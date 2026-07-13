from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_web = _module("ghostline_build_web", ROOT / "scripts" / "build_web.py")
web_runtime = _module("ghostline_web_runtime", ROOT / "web" / "runtime.py")


def test_web_observation_serialization_is_player_equivalent() -> None:
    shapes = {
        "ego": (24,),
        "objective": (8,),
        "local_grid": (8, 15, 15),
        "targets": (5, 10),
        "target_mask": (5,),
        "entities": (12, 13),
        "entity_mask": (12,),
        "rays": (24, 3),
        "action_mask": (36,),
    }
    observation = {key: np.zeros(shape, dtype=np.float32) for key, shape in shapes.items()}
    payload = json.loads(web_runtime.observation_json(observation))
    assert set(payload) == set(web_runtime.OBSERVATION_KEYS)
    assert len(payload["objective"]) == 8
    assert len(payload["local_grid"]) == 8
    with pytest.raises(KeyError, match="objective"):
        web_runtime.observation_json({key: value for key, value in observation.items() if key != "objective"})


def test_policy_manifest_supports_v2_objective_and_human_fallback(tmp_path: Path) -> None:
    manifest = build_web._write_policy_manifest(tmp_path, None)
    assert manifest["available"] is False
    assert manifest["model_url"] is None
    assert manifest["inputs"]["objective"] == [1, 8]
    assert manifest["inputs"]["hidden"] is None
    assert manifest["hidden_size"] is None
    assert manifest["runtime"] == "onnxruntime-web@1.27.0"


def test_policy_manifest_derives_recurrent_width_from_onnx(tmp_path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper

    inputs = []
    for name, shape in {**build_web.POLICY_INPUT_SHAPES, "hidden": [1, 1, 256]}.items():
        dtype = TensorProto.INT8 if name.endswith("mask") else TensorProto.FLOAT
        inputs.append(helper.make_tensor_value_info(name, dtype, shape))
    outputs = [
        helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 36]),
        helper.make_tensor_value_info("value", TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info("next_hidden", TensorProto.FLOAT, [1, 1, 256]),
    ]
    nodes = [
        helper.make_node(
            "Constant",
            [],
            ["logits"],
            value=helper.make_tensor("logits_value", TensorProto.FLOAT, [1, 36], [0.0] * 36),
        ),
        helper.make_node(
            "Constant",
            [],
            ["value"],
            value=helper.make_tensor("value_value", TensorProto.FLOAT, [1], [0.0]),
        ),
        helper.make_node("Identity", ["hidden"], ["next_hidden"]),
    ]
    graph = helper.make_graph(nodes, "policy", inputs, outputs)
    model = helper.make_model(graph)
    model.metadata_props.add(key="ghostline.contract", value="GhostlineEnv-v2")
    fingerprint = build_web._current_environment_fingerprint()
    model.metadata_props.add(key="ghostline.environment_fingerprint", value=fingerprint)
    model_path = tmp_path / "policy.onnx"
    onnx.save(model, model_path)

    static = tmp_path / "static"
    static.mkdir()
    manifest = build_web._write_policy_manifest(static, model_path)
    assert manifest["hidden_size"] == 256
    assert manifest["inputs"]["hidden"] == [1, 1, 256]
    assert manifest["model_metadata"] == {
        "ghostline.contract": "GhostlineEnv-v2",
        "ghostline.environment_fingerprint": fingerprint,
    }


def test_policy_contract_rejects_stale_environment_metadata(tmp_path: Path) -> None:
    import onnx
    from onnx import TensorProto, helper

    inputs = [
        helper.make_tensor_value_info(
            name,
            TensorProto.INT8 if name.endswith("mask") else TensorProto.FLOAT,
            shape,
        )
        for name, shape in {**build_web.POLICY_INPUT_SHAPES, "hidden": [1, 1, 256]}.items()
    ]
    outputs = [
        helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 36]),
        helper.make_tensor_value_info("value", TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info("next_hidden", TensorProto.FLOAT, [1, 1, 256]),
    ]
    model = helper.make_model(helper.make_graph([], "stale-policy", inputs, outputs))
    model.metadata_props.add(key="ghostline.contract", value="GhostlineEnv-v2")
    model.metadata_props.add(key="ghostline.environment_fingerprint", value="stale")
    path = tmp_path / "stale.onnx"
    onnx.save(model, path)

    with pytest.raises(RuntimeError, match="current frozen environment fingerprint"):
        build_web._onnx_policy_contract(path)


def test_portfolio_web_build_requires_model_unless_explicitly_human_only(monkeypatch, capsys) -> None:
    monkeypatch.setattr(build_web, "_resolve_model", lambda requested: None)
    assert build_web.build() == 2
    assert "portfolio web builds require" in capsys.readouterr().err


def test_browser_progression_round_trip_and_invalid_data_fallback(tmp_path: Path) -> None:
    class Storage:
        def __init__(self):
            self.values: dict[str, str] = {}

        def getItem(self, key: str):
            return self.values.get(key)

        def setItem(self, key: str, value: str) -> None:
            self.values[key] = value

    class Host:
        localStorage = Storage()

    path = tmp_path / "progression-v1.json"
    path.write_text(json.dumps({"version": 1, "highest_unlocked_tier": 3}), encoding="utf-8")
    assert web_runtime.persist_progression(Host, path)
    path.unlink()
    assert web_runtime.hydrate_progression(Host, path)
    assert json.loads(path.read_text(encoding="utf-8"))["highest_unlocked_tier"] == 3
    Host.localStorage.values[web_runtime.PROGRESSION_STORAGE_KEY] = "not-json"
    assert web_runtime.hydrate_progression(Host, path) is False


def test_browser_policy_prefetch_uses_completed_action_without_duplicate_inference() -> None:
    class Bridge:
        def __init__(self):
            self.steps = 0

        def step(self, payload: str) -> int:
            assert json.loads(payload)["ego"] == [0.0] * 24
            self.steps += 1
            return 0

        def currentAction(self) -> int:
            return 17

    class Host:
        ghostlinePolicy = Bridge()

    observation = {
        "ego": np.zeros(24, dtype=np.float32),
        "objective": np.zeros(8, dtype=np.float32),
        "local_grid": np.zeros((8, 15, 15), dtype=np.float32),
        "targets": np.zeros((5, 10), dtype=np.float32),
        "target_mask": np.zeros(5, dtype=np.int8),
        "entities": np.zeros((12, 13), dtype=np.float32),
        "entity_mask": np.zeros(12, dtype=np.int8),
        "rays": np.zeros((24, 3), dtype=np.float32),
        "action_mask": np.ones(36, dtype=np.int8),
    }
    policy = web_runtime.BrowserOnnxPolicy(Host)
    policy.prefetch(observation)
    action, hidden = policy.act(observation)
    assert (action, hidden) == (17, None)
    assert Host.ghostlinePolicy.steps == 1


def test_mixed_control_run_is_marked_hybrid_and_releases_agent_wrapper() -> None:
    class Environment:
        def __init__(self):
            self.closed = False

        def close(self) -> None:
            self.closed = True

    class Simulation:
        elapsed_seconds = 12.5
        data = 3

    class App:
        sim = Simulation()
        agent_env = Environment()
        agent_observation = object()
        agent_hidden = object()
        _telemetry = {"controller": "human", "policy": "keyboard"}

    class Host:
        ghostlinePolicy = object()

    runtime = web_runtime.GhostlineWebRuntime(App(), host=Host())
    environment = runtime.app.agent_env
    runtime._mark_hybrid_run()
    runtime._release_agent_environment()

    assert runtime.run_mode == "hybrid"
    assert runtime.app._telemetry["controller"] == "hybrid"
    assert runtime.app._telemetry["takeover_elapsed_seconds"] == 12.5
    assert runtime.app._telemetry["takeover_data"] == 3
    assert environment.closed is True
    assert runtime.app.agent_env is None


def test_live_policy_failure_restores_human_control_without_stale_action() -> None:
    class Environment:
        closed = False

        def close(self) -> None:
            self.closed = True

    class Simulation:
        elapsed_seconds = 9.25
        data = 1
        terminated = False
        truncated = False

    class Bridge:
        resets = 0

        def reset(self) -> None:
            self.resets += 1

    class Shell:
        mode = "agent"
        state = None
        notice = None

        def setControlMode(self, mode: str) -> None:
            self.mode = mode

        def setPolicyState(self, state: str, message: str) -> None:
            self.state = (state, message)

        def showNotice(self, message: str, kind: str) -> None:
            self.notice = (message, kind)

    class Host:
        ghostlinePolicy = Bridge()
        ghostlineShell = Shell()

    class App:
        state = "lab_play"
        sim = Simulation()
        agent_env = Environment()
        agent_observation = object()
        agent_hidden = object()
        learned_policy = object()
        _agent_action = 23
        _telemetry = {"controller": "agent", "policy": "recurrent policy"}

    runtime = web_runtime.GhostlineWebRuntime(App(), host=Host())
    runtime.control_mode = "agent"
    runtime.run_mode = "agent"
    runtime.policy.prefetched = True
    environment = runtime.app.agent_env
    runtime._restore_human_after_policy_failure()

    assert runtime.control_mode == "human"
    assert runtime.run_mode == "hybrid"
    assert runtime.app.state == "play"
    assert runtime.app._agent_action == 0
    assert runtime.app.learned_policy is None
    assert runtime.policy.prefetched is False
    assert runtime.app._telemetry["controller"] == "hybrid"
    assert environment.closed is True
    assert runtime.app.agent_env is None
    assert Host.ghostlinePolicy.resets == 1
    assert Host.ghostlineShell.mode == "human"
    assert Host.ghostlineShell.state[0] == "unavailable"
    assert "manual control" in Host.ghostlineShell.notice[0]


def _fake_pygbag_runtime(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = b"runtime"
    fake_hash = hashlib.sha256(payload).hexdigest()
    monkeypatch.setattr(
        build_web,
        "PYGBAG_RUNTIME_PUBLISHED_SHA256",
        {relative: fake_hash for relative in build_web.PYGBAG_RUNTIME_PUBLISHED_SHA256},
    )
    monkeypatch.setattr(build_web, "PYGBAG_LICENSE_SHA256", hashlib.sha256(b"license").hexdigest())
    monkeypatch.setattr(build_web, "CPYTHON_LICENSE_SHA256", hashlib.sha256(b"license").hexdigest())
    for relative in build_web.PYGBAG_RUNTIME_PUBLISHED_SHA256:
        destination = root / build_web.PYGBAG_RUNTIME_ROOT / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
    for relative in (build_web.PYGBAG_LICENSE_PATH, build_web.CPYTHON_LICENSE_PATH):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"license")


def _fake_bundle(root: Path, monkeypatch: pytest.MonkeyPatch, *, with_model: bool = False) -> None:
    for name in (
        "index.html",
        "ghostline.tar.gz",
        "embed-bridge.mjs",
        "ghostline-shell.mjs",
        "matched-runs.mjs",
        "policy-bridge.mjs",
        "ghostline.css",
        "ghostline-key-art.webp",
        "favicon.png",
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
    ):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(b"release")
    runtime_url = f"./{build_web.PYGBAG_RUNTIME_ROOT.as_posix()}/"
    (root / "index.html").write_text(
        f'<script src="{runtime_url}pythons.js" data-os="snd,gui"></script>'
        f'<script>config = {{cdn: "{runtime_url}"}};</script>',
        encoding="utf-8",
    )
    _fake_pygbag_runtime(root, monkeypatch)
    browserfs = root / build_web.BROWSERFS_WEB_PATH
    browserfs.parent.mkdir(parents=True, exist_ok=True)
    browserfs.write_bytes(b"browserfs")
    browserfs_license = root / build_web.BROWSERFS_LICENSE_PATH
    browserfs_license.parent.mkdir(parents=True, exist_ok=True)
    browserfs_license.write_bytes(b"MIT")
    manifest = {"available": False, "model_url": None}
    if with_model:
        relative = Path("models/ghostline-policy-deadbeef.onnx")
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_bytes(b"model")
        manifest = {"available": True, "model_url": relative.as_posix()}
        vendor = root / build_web.ORT_WEB_ROOT
        vendor.mkdir(parents=True, exist_ok=True)
        for name in (
            "ort.all.min.mjs",
            "ort-wasm-simd-threaded.mjs",
            "ort-wasm-simd-threaded.wasm",
        ):
            (vendor / name).write_bytes(b"runtime")
        for relative in (build_web.ORT_LICENSE_PATH, build_web.ORT_NOTICES_PATH):
            destination = root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"release terms")
    (root / "policy-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_bundle_budget_report_distinguishes_lazy_agent_payload(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_bundle(tmp_path, monkeypatch, with_model=True)
    report = build_web.validate_bundle(tmp_path, initial_budget=1024, wasm_agent_budget=2048)
    assert report["model_available"] is True
    assert report["wasm_agent_total_bytes_local"] > report["human_first_run_bytes_local"]
    assert report["runtime_module_allowlist"] == list(build_web.WEB_RUNTIME_MODULES)
    assert {item["path"] for item in report["legal_documents"]} == {
        "THIRD_PARTY_NOTICES.md",
        build_web.PYGBAG_LICENSE_PATH.as_posix(),
        build_web.CPYTHON_LICENSE_PATH.as_posix(),
        build_web.BROWSERFS_LICENSE_PATH.as_posix(),
        build_web.ORT_LICENSE_PATH.as_posix(),
        build_web.ORT_NOTICES_PATH.as_posix(),
    }
    assert (tmp_path / "bundle-report.json").is_file()
    with pytest.raises(RuntimeError, match="Human first-run payload"):
        build_web.validate_bundle(tmp_path, initial_budget=1, wasm_agent_budget=2048)


def test_agent_bundle_rejects_missing_onnx_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_bundle(tmp_path, monkeypatch, with_model=True)
    (tmp_path / build_web.ORT_WEB_ROOT / "ort-wasm-simd-threaded.wasm").unlink()
    with pytest.raises(RuntimeError, match="missing WASM runtime"):
        build_web.bundle_report(tmp_path)

    _fake_bundle(tmp_path, monkeypatch, with_model=True)
    (tmp_path / build_web.ORT_NOTICES_PATH).unlink()
    with pytest.raises(RuntimeError, match="ThirdPartyNotices"):
        build_web.bundle_report(tmp_path)


def test_web_stage_is_an_explicit_runtime_and_asset_allowlist(monkeypatch, tmp_path: Path) -> None:
    stage = tmp_path / "ghostline-stage"
    monkeypatch.setattr(build_web, "STAGE", stage)

    def fake_browserfs(static: Path) -> None:
        script = static / build_web.BROWSERFS_WEB_PATH
        license_file = static / build_web.BROWSERFS_LICENSE_PATH
        script.parent.mkdir(parents=True, exist_ok=True)
        license_file.parent.mkdir(parents=True, exist_ok=True)
        script.write_bytes(b"browserfs")
        license_file.write_bytes(b"MIT")

    def fake_pygbag(static: Path) -> None:
        for relative in build_web.PYGBAG_RUNTIME_PUBLISHED_SHA256:
            destination = static / build_web.PYGBAG_RUNTIME_ROOT / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"runtime")
        for relative in (build_web.PYGBAG_LICENSE_PATH, build_web.CPYTHON_LICENSE_PATH):
            destination = static / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"license")

    monkeypatch.setattr(build_web, "_stage_browserfs", fake_browserfs)
    monkeypatch.setattr(build_web, "_stage_pygbag_runtime", fake_pygbag)
    build_web.stage(model=None, include_ort=False)

    staged_modules = {path.name for path in (stage / "ghostline").glob("*.py")}
    assert staged_modules == set(build_web.WEB_RUNTIME_MODULES)
    assert not staged_modules & {
        "ablation.py",
        "cli.py",
        "evaluation.py",
        "exporting.py",
        "imitation.py",
        "model.py",
        "packaging.py",
        "recording.py",
        "torchrl_train.py",
        "training.py",
    }
    staged_assets = {
        path.relative_to(stage).as_posix()
        for path in (stage / "assets").rglob("*")
        if path.is_file()
    }
    expected_assets = {
        "assets/licenses.json",
        *{
            path.relative_to(ROOT).as_posix()
            for path in build_web._runtime_asset_paths(ROOT)
        },
    }
    assert staged_assets == expected_assets
    assert not any("screenshots" in name or "source" in name for name in staged_assets)
    assert (stage / "static" / "THIRD_PARTY_NOTICES.md").is_file()
    assert (stage / "static" / "embed-bridge.mjs").is_file()
    assert (stage / "static" / build_web.PYGBAG_RUNTIME_ROOT / "cpython312/main.wasm").is_file()


def test_web_asset_allowlist_rejects_provenance_and_screenshot_files(tmp_path: Path) -> None:
    source = tmp_path / "assets" / "screenshots" / "portfolio-source.png"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"not runtime art")
    (tmp_path / "assets" / "licenses.json").write_text(
        json.dumps(
            {
                "project": "Ghostline",
                "runtime_distribution": {
                    "license": "MIT",
                    "files": ["assets/screenshots/portfolio-source.png"],
                },
                "visual_assets": {
                    "bad": {"runtime_file": "assets/screenshots/portfolio-source.png"}
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="cannot be a web runtime asset"):
        build_web._runtime_asset_paths(tmp_path)


def test_web_shell_and_policy_bridge_include_release_behaviors() -> None:
    template = (ROOT / "web" / "ghostline.tmpl").read_text(encoding="utf-8")
    shell = (ROOT / "web" / "static" / "ghostline-shell.mjs").read_text(encoding="utf-8")
    embed = (ROOT / "web" / "static" / "embed-bridge.mjs").read_text(encoding="utf-8")
    policy = (ROOT / "web" / "static" / "policy-bridge.mjs").read_text(encoding="utf-8")
    for element in ("launch-gate", "agent-control", "human-control", "tier-select", "seed-input", "fullscreen-control"):
        assert f'id="{element}"' in template
    assert "AGENT TAKEOVER" in template
    assert 'get("embed") === "1"' in template
    assert "consumeCommand" in shell
    assert "new GhostlineEmbedBridge()" in shell
    assert "publishRunComplete(metrics)" in shell
    assert "modelAvailable" in embed
    assert 'type: "run-complete"' in embed
    assert 'query.get("autoplay")' in shell
    manifest_listener = shell.split(
        'addEventListener("ghostline:policy-manifest"', 1
    )[1].split('addEventListener("ghostline:policy-state"', 1)[0]
    assert "maybePublishEmbedReady()" in manifest_listener
    assert "if (policyAvailability === null) return" in shell
    assert template.index("while not platform.window.MM.UME") < template.index("await shell.source(main")
    assert 'ghostlineShell.setBootState("ready")' not in template
    assert "self.host.ghostlineShell.markGameReady()" in (
        ROOT / "web" / "runtime.py"
    ).read_text(encoding="utf-8")
    assert 'executionProviders: ["webgpu", "wasm"]' in policy
    assert 'executionProviders: ["wasm"]' in policy
    assert "results.next_hidden" in policy
    assert "pendingObservation" in policy
    assert 'queue("pause-hidden")' in shell
    assert 'queue("pause-focus")' in shell
    assert 'metrics.mode === "hybrid"' in shell
    assert 'queue("policy-failed")' in shell
    assert 'matchedRunSnapshot(runHistory)' in shell
    assert '$("seed-input").value = String(metrics.seed)' in shell
    assert 'id="match-status"' in template
    assert 'href="./THIRD_PARTY_NOTICES.md"' in template


def test_browser_gymnasium_shim_supports_env_seeding_and_spaces() -> None:
    shim = web_runtime._browser_gymnasium_shim()
    env = shim.Env()
    env.reset(seed=17)
    assert int(env.np_random.integers(0, 1000)) == 740
    discrete = shim.spaces.Discrete(36)
    box = shim.spaces.Box(-1.0, 1.0, shape=(2, 3), dtype=np.float32)
    dictionary = shim.spaces.Dict({"box": box})
    assert discrete.n == 36
    assert box.shape == (2, 3)
    assert box.dtype == np.dtype(np.float32)
    assert dictionary.spaces == {"box": box}


def test_bundle_rejects_external_or_modified_pygbag_runtime(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _fake_bundle(tmp_path, monkeypatch)
    (tmp_path / "index.html").write_text(
        '<script src="https://pygame-web.github.io/cdn/0.9.3/pythons.js"></script>',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="external Pygbag runtime CDN"):
        build_web.bundle_report(tmp_path)

    _fake_bundle(tmp_path, monkeypatch)
    runtime_file = tmp_path / build_web.PYGBAG_RUNTIME_ROOT / "cpython312/main.wasm"
    runtime_file.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="runtime hash mismatch"):
        build_web.bundle_report(tmp_path)


def test_vercel_headers_enable_threaded_wasm_and_immutable_models() -> None:
    config = json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))
    assert config["outputDirectory"] == ".web-build/ghostline/build/web"
    assert "--model models/ghostline-policy.onnx" in config["buildCommand"]
    assert ".[web]" in config["buildCommand"]
    headers = config["headers"]
    global_headers = {item["key"]: item["value"] for item in headers[0]["headers"]}
    assert global_headers["Cross-Origin-Opener-Policy"] == "same-origin"
    # The CPython bootstrap is self-hosted; credentialless remains the tested
    # isolation mode for the standalone threaded-WASM path.
    assert global_headers["Cross-Origin-Embedder-Policy"] == "credentialless"
    immutable_sources = {
        entry["source"]
        for entry in headers
        if any(header.get("value", "").endswith("immutable") for header in entry["headers"])
    }
    assert "/vendor/(.*)" in immutable_sources
    assert "/runtime/(.*)" in immutable_sources
    assert "/models/(.*)" in immutable_sources
