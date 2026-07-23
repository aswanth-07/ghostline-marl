from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest

from ghostline.cli import build_parser
from ghostline.packaging import (
    ASSET_MANIFEST,
    DEFAULT_POLICY,
    EXCLUDED_PLAYER_MODULES,
    MIN_RELEASE_PARITY_SAMPLES,
    POLICY_PARITY_DOCUMENT,
    THIRD_PARTY_NOTICES,
    _canonical_policy_source,
    _release_assets,
    _runtime_asset_paths,
    _release_policy_evidence,
    _verify_player_smoke,
    _verify_player_archive,
    _write_manifest,
    build_windows,
    windows_build_command,
)
from ghostline.onnx_contract import POLICY_INPUT_SHAPES, environment_fingerprint


ROOT = Path(__file__).resolve().parents[1]


RUNTIME_ASSET_NAMES = (
    "ghostline-environment-atlas-v1.png",
    "ghostline-character-security-atlas-v1.png",
    "ghostline-diagonal-locomotion-v2.png",
)


def test_bundled_onnx_policy_clears_every_disclosed_watch_agent_seed() -> None:
    from ghostline.app import AGENT_SHOWCASE_SEEDS
    from ghostline.env import GhostlineEnv
    from ghostline.inference import OnnxGhostlinePolicy

    policy = OnnxGhostlinePolicy(ROOT / "models" / "ghostline-policy.onnx")
    for tier, seed in AGENT_SHOWCASE_SEEDS.items():
        env = GhostlineEnv(seed=seed, tier=tier)
        try:
            observation, _ = env.reset(seed=seed)
            hidden = None
            terminated = truncated = False
            info = {}
            while not (terminated or truncated):
                action, hidden = policy.act(observation, hidden, deterministic=True)
                observation, _, terminated, truncated, info = env.step(action)
            assert info["is_success"], f"tier {tier} seed {seed}: {info['fail_reason']}"
        finally:
            env.close()


def test_portfolio_video_contract_replays_the_bundled_onnx_action_sequence() -> None:
    """Bind the recorded showcase to the same graph used by live takeover."""

    from ghostline.app import PORTFOLIO_DEMO_SEED, PORTFOLIO_DEMO_TIER
    from ghostline.env import GhostlineEnv
    from ghostline.inference import OnnxGhostlinePolicy

    env = GhostlineEnv(seed=PORTFOLIO_DEMO_SEED, tier=PORTFOLIO_DEMO_TIER)
    policy = OnnxGhostlinePolicy(ROOT / "models" / "ghostline-policy.onnx")
    try:
        observation, _ = env.reset(seed=PORTFOLIO_DEMO_SEED)
        hidden = None
        actions = bytearray()
        terminated = truncated = False
        info = {}
        while not (terminated or truncated):
            action, hidden = policy.act(observation, hidden, deterministic=True)
            actions.append(action)
            observation, _, terminated, truncated, info = env.step(action)
    finally:
        env.close()

    assert info["is_success"] is True
    assert len(actions) == 366
    assert info["duration_seconds"] == pytest.approx(36.5333333333)
    assert info["damage"] == 2
    assert hashlib.sha256(actions).hexdigest() == "7887d2fba31b6aeac5e7c4462c2258d28aec428182ec2e906e8450b314591925"


def _write_release_inputs(root: Path) -> None:
    (root / "src" / "ghostline").mkdir(parents=True, exist_ok=True)
    (root / "src" / "ghostline" / "player_entry.py").write_text("", encoding="utf-8")
    (root / "LICENSE").write_text("MIT License\n", encoding="utf-8")
    (root / THIRD_PARTY_NOTICES).write_text("# notices\n", encoding="utf-8")
    visual = root / "assets" / "visual"
    visual.mkdir(parents=True, exist_ok=True)
    runtime_files: list[str] = []
    for name in RUNTIME_ASSET_NAMES:
        relative = Path("assets/visual") / name
        (root / relative).write_bytes(name.encode("utf-8"))
        runtime_files.append(relative.as_posix())
    (root / ASSET_MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": 2,
                "project": "Ghostline",
                "runtime_distribution": {"license": "MIT", "files": runtime_files},
            }
        ),
        encoding="utf-8",
    )


def _write_test_onnx_policy(path: Path, *, fingerprint: str) -> None:
    import onnx
    from onnx import TensorProto, helper

    inputs = [
        helper.make_tensor_value_info(
            name,
            TensorProto.INT8 if name.endswith("mask") else TensorProto.FLOAT,
            shape,
        )
        for name, shape in {**POLICY_INPUT_SHAPES, "hidden": [1, 1, 256]}.items()
    ]
    outputs = [
        helper.make_tensor_value_info("logits", TensorProto.FLOAT, [1, 36]),
        helper.make_tensor_value_info("value", TensorProto.FLOAT, [1]),
        helper.make_tensor_value_info("next_hidden", TensorProto.FLOAT, [1, 1, 256]),
    ]
    model = helper.make_model(helper.make_graph([], "release-policy", inputs, outputs))
    model.metadata_props.add(key="ghostline.contract", value="GhostlineEnv-v2")
    model.metadata_props.add(key="ghostline.environment_fingerprint", value=fingerprint)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)


def _write_parity_report(model: Path, *, fingerprint: str, mismatches: int = 0) -> Path:
    model_sha256 = hashlib.sha256(model.read_bytes()).hexdigest()
    report = {
        "report_version": 2,
        "status": "passed",
        "checkpoint_sha256": "a" * 64,
        "parity_samples": MIN_RELEASE_PARITY_SAMPLES,
        "mismatches": mismatches,
        "observation_contract": "GhostlineEnv-v2",
        "environment_fingerprint": fingerprint,
        "artifacts": {
            "fp32": {
                "sha256": model_sha256,
                "precision": "fp32",
                "parity": {
                    "passed": mismatches == 0,
                    "samples": MIN_RELEASE_PARITY_SAMPLES,
                    "action_mismatches": mismatches,
                    "sequence_horizon": 128,
                    "tiers": [1, 2, 3, 4, 5, 6],
                },
            }
        },
    }
    path = model.with_suffix(".parity.json")
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def test_base_imports_are_headless_and_training_free() -> None:
    probe = """
import sys
import ghostline
from ghostline.generation import LevelGenerator
from ghostline.simulation import GhostlineSimulation
assert 'pygame' not in sys.modules
assert 'torch' not in sys.modules
level = LevelGenerator().generate(seed=10101, tier=1)
sim = GhostlineSimulation(seed=10101, tier=1)
assert level.seed == sim.seed == 10101
"""
    subprocess.run([sys.executable, "-c", probe], cwd=ROOT, check=True)


def test_player_entry_has_a_headless_human_only_release_smoke() -> None:
    subprocess.run(
        [sys.executable, "-m", "ghostline.player_entry", "--release-smoke-test", "--human-only"],
        cwd=ROOT,
        check=True,
    )


def test_windows_command_bundles_policy_and_excludes_training_stack(tmp_path: Path) -> None:
    root = tmp_path
    _write_release_inputs(root)
    visual = root / "assets" / "visual"
    (visual / "menu.png").write_bytes(b"menu")
    (visual / "concept-source.png").write_bytes(b"source")
    (visual / "menu-web.webp").write_bytes(b"web")
    screenshots = root / "assets" / "screenshots"
    screenshots.mkdir()
    (screenshots / "portfolio.png").write_bytes(b"screenshot")
    policy = root / DEFAULT_POLICY
    policy.parent.mkdir()
    policy.write_bytes(b"onnx")

    command = windows_build_command(root=root, model=policy)
    joined = " ".join(command)
    assert str(root / "src" / "ghostline" / "player_entry.py") == command[-1]
    assert "ghostline/__main__.py" not in joined.replace("\\", "/")
    assert "ghostline.inference" in command
    assert "--collect-all onnxruntime" not in joined
    assert f"{policy}{os.pathsep}models" in command
    for name in RUNTIME_ASSET_NAMES:
        assert f"{visual / name}{os.pathsep}assets/visual" in command
    assert f"{root / ASSET_MANIFEST}{os.pathsep}assets" in command
    assert f"{root / 'LICENSE'}{os.pathsep}." in command
    assert f"{root / THIRD_PARTY_NOTICES}{os.pathsep}." in command
    assert "menu.png" not in joined
    assert "concept-source.png" not in joined
    assert "menu-web.webp" not in joined
    assert "portfolio.png" not in joined
    for module in EXCLUDED_PLAYER_MODULES:
        assert module in command


def test_package_requires_policy_unless_human_only(tmp_path: Path, capsys) -> None:
    _write_release_inputs(tmp_path)
    assert build_windows(root=tmp_path, model=Path("models/missing.onnx"), dry_run=True) == 2
    assert "selected ONNX policy does not exist" in capsys.readouterr().err
    assert build_windows(root=tmp_path, model=None, dry_run=True) == 0
    output = capsys.readouterr().out
    assert "player_entry.py" in output
    assert "--collect-all onnxruntime" not in output
    assert "--exclude-module onnxruntime" in output


def test_package_manifest_hashes_executable_and_policy(tmp_path: Path) -> None:
    _write_release_inputs(tmp_path)
    executable = tmp_path / "dist" / "Ghostline.exe"
    executable.parent.mkdir()
    executable.write_bytes(b"windows-player")
    policy = tmp_path / "policy.onnx"
    policy.write_bytes(b"selected-policy")
    policy_sha256 = hashlib.sha256(policy.read_bytes()).hexdigest()
    validation = {
        "contract": "GhostlineEnv-v2",
        "environment_fingerprint": "f" * 64,
        "recurrent_size": 256,
        "onnx_sha256": policy_sha256,
        "checkpoint_sha256": "a" * 64,
        "precision": "fp32",
        "parity_samples": MIN_RELEASE_PARITY_SAMPLES,
        "action_mismatches": 0,
        "sequence_horizon": 128,
        "tiers": [1, 2, 3, 4, 5, 6],
        "parity_report_file": POLICY_PARITY_DOCUMENT,
        "parity_report_sha256": "b" * 64,
    }
    manifest_path = _write_manifest(
        tmp_path,
        executable,
        policy,
        policy_validation=validation,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == 3
    assert manifest["policy_bundled"] is True
    assert manifest["policy_file"] == "models/ghostline-policy.onnx"
    assert manifest["pytorch_bundled"] is False
    assert "torch" in manifest["forbidden_package_roots_verified_absent"]
    assert len(manifest["executable_sha256"]) == 64
    assert len(manifest["policy_sha256"]) == 64
    assert manifest["policy_validation"] == validation
    assert manifest["license"] == "MIT"
    assert len(manifest["license_sha256"]) == 64
    assert manifest["asset_manifest"]["file"] == "ASSET-LICENSES.json"
    assert {item["file"] for item in manifest["runtime_assets"]} == {
        f"assets/visual/{name}" for name in RUNTIME_ASSET_NAMES
    }
    assert manifest["user_data"] == {
        "root": "%LOCALAPPDATA%/Ghostline",
        "progression": "progression-v1.json",
        "run_telemetry": "runs-v1.jsonl",
    }
    assert manifest["recording"]["run_telemetry_enabled"] is True
    assert manifest["recording"]["video_recorder_bundled"] is False


def test_portfolio_policy_requires_current_metadata_and_exact_parity_evidence(tmp_path: Path) -> None:
    fingerprint = environment_fingerprint(ROOT / "src" / "ghostline")
    policy = tmp_path / "ghostline-policy.onnx"
    _write_test_onnx_policy(policy, fingerprint=fingerprint)
    parity = _write_parity_report(policy, fingerprint=fingerprint)

    evidence, selected_report = _release_policy_evidence(ROOT, policy)
    assert selected_report == parity.resolve()
    assert evidence["contract"] == "GhostlineEnv-v2"
    assert evidence["environment_fingerprint"] == fingerprint
    assert evidence["onnx_sha256"] == hashlib.sha256(policy.read_bytes()).hexdigest()
    assert evidence["checkpoint_sha256"] == "a" * 64
    assert evidence["parity_samples"] == MIN_RELEASE_PARITY_SAMPLES
    assert evidence["action_mismatches"] == 0

    parity.unlink()
    with pytest.raises(RuntimeError, match="no export parity report"):
        _release_policy_evidence(ROOT, policy)


def test_portfolio_policy_rejects_stale_or_mismatched_parity(tmp_path: Path) -> None:
    fingerprint = environment_fingerprint(ROOT / "src" / "ghostline")
    policy = tmp_path / "ghostline-policy.onnx"
    _write_test_onnx_policy(policy, fingerprint=fingerprint)
    _write_parity_report(policy, fingerprint=fingerprint, mismatches=1)
    with pytest.raises(RuntimeError, match="zero mismatches"):
        _release_policy_evidence(ROOT, policy)

    policy.unlink()
    _write_test_onnx_policy(policy, fingerprint="stale")
    with pytest.raises(RuntimeError, match="current frozen environment fingerprint"):
        _release_policy_evidence(ROOT, policy)


def test_arbitrary_policy_path_is_staged_under_runtime_filename(tmp_path: Path) -> None:
    source = tmp_path / "experiment" / "policy.onnx"
    source.parent.mkdir()
    source.write_bytes(b"checkpoint")
    staged = _canonical_policy_source(tmp_path, source)
    assert staged == tmp_path / "build" / "Ghostline-package" / DEFAULT_POLICY.name
    assert staged.read_bytes() == source.read_bytes()


def test_archive_verifier_rejects_training_packages(monkeypatch, tmp_path: Path) -> None:
    class Result:
        returncode = 0
        stdout = " 1, 2, 3, 1, 'm', 'ghostline.app'\n 1, 2, 3, 1, 'm', 'torch.nn'\n"

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: Result())
    with pytest.raises(RuntimeError, match="forbidden packages: torch"):
        _verify_player_archive(tmp_path / "Ghostline.exe")


def test_asset_manifest_is_the_only_runtime_asset_authority(tmp_path: Path) -> None:
    _write_release_inputs(tmp_path)
    screenshots = tmp_path / "assets" / "screenshots"
    screenshots.mkdir()
    (screenshots / "portfolio.png").write_bytes(b"not-runtime")
    (tmp_path / "assets" / "visual" / "draft-source.png").write_bytes(b"not-runtime")

    declared = _runtime_asset_paths(tmp_path)
    release = _release_assets(tmp_path)
    assert {path.name for path in declared} == set(RUNTIME_ASSET_NAMES)
    assert {path.name for path, _ in release} == {"licenses.json", *RUNTIME_ASSET_NAMES}

    manifest = json.loads((tmp_path / ASSET_MANIFEST).read_text(encoding="utf-8"))
    manifest["runtime_distribution"]["files"] = ["../outside.png"]
    (tmp_path / ASSET_MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="escapes assets"):
        _runtime_asset_paths(tmp_path)


def test_packaged_smoke_uses_policy_gate_and_reports_failure(monkeypatch, tmp_path: Path) -> None:
    calls: list[tuple[list[str], int]] = []

    class Result:
        returncode = 0

    def successful(command, **kwargs):
        calls.append((list(command), int(kwargs["timeout"])))
        return Result()

    executable = tmp_path / "Ghostline.exe"
    monkeypatch.setattr(subprocess, "run", successful)
    _verify_player_smoke(executable, policy_bundled=True)
    _verify_player_smoke(executable, policy_bundled=False)
    assert calls[0] == ([str(executable), "--release-smoke-test"], 120)
    assert calls[1] == ([str(executable), "--release-smoke-test", "--human-only"], 120)

    Result.returncode = 7
    with pytest.raises(RuntimeError, match="exit code 7"):
        _verify_player_smoke(executable, policy_bundled=True)


def test_release_user_data_stays_under_localappdata(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    from ghostline.progression import (
        progression_path,
        record_run,
        save_settings,
        telemetry_path,
    )

    save_settings({"audio": {"master": 0.4}})
    record_run({"controller": "release-test", "seed": 8, "tier": 1, "success": True})
    expected_root = tmp_path / "Ghostline"
    assert progression_path() == expected_root / "progression-v1.json"
    assert telemetry_path() == expected_root / "runs-v1.jsonl"
    assert progression_path().is_file() and telemetry_path().is_file()
    assert not (tmp_path / "progression-v1.json").exists()


def test_public_cli_and_extras_match_release_contract() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as version_exit:
        parser.parse_args(["--version"])
    assert version_exit.value.code == 0
    package = parser.parse_args(["package"])
    assert package.model == DEFAULT_POLICY
    assert package.human_only is False
    assert parser.parse_args(["package", "--human-only"]).human_only is True

    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert project["project"]["scripts"] == {"ghostline": "ghostline.cli:main"}
    assert project["project"]["urls"]["Repository"] == "https://github.com/aswanth-07/ghostline"
    assert project["tool"]["setuptools"]["packages"]["find"]["include"] == ["ghostline*"]
    assert project["tool"]["pytest"]["ini_options"]["pythonpath"] == ["src"]
    assert project["project"]["optional-dependencies"]["agent"] == ["onnxruntime==1.27.0"]
    assert "onnx==1.22.0" in project["project"]["optional-dependencies"]["dev"]
    assert "onnxruntime==1.27.0" in project["project"]["optional-dependencies"]["dev"]
    assert "torch==2.13.0" not in project["project"]["optional-dependencies"]["build"]
    assert "onnxruntime==1.27.0" in project["project"]["optional-dependencies"]["build"]
    assert project["project"]["license"] == "MIT"
    assert not any(value.startswith("License ::") for value in project["project"]["classifiers"])
    assert project["project"]["license-files"] == ["LICENSE", "THIRD_PARTY_NOTICES.md"]
    source_manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "prune src/neon_arena" in source_manifest
    assert "include THIRD_PARTY_NOTICES.md" in source_manifest
    assert "include requirements.lock" in source_manifest
    assert "include models/model-card.md" in source_manifest
    assert "recursive-include assets *" in source_manifest
    assert "recursive-include benchmarks" in source_manifest
    assert "recursive-include scripts *.py" in source_manifest
    assert "recursive-include web *" in source_manifest
    assert "recursive-include wiki *.md" in source_manifest
    assert "recursive-include tests *.py" in source_manifest
    assert "exclude tests/test_cli.py" in source_manifest
    assert "exclude tests/test_env.py" in source_manifest
    assert "exclude tests/test_training.py" in source_manifest


def test_workflows_use_locked_installs_and_release_smoke() -> None:
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    release = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    assert "--constraint requirements.lock" in ci
    assert "branches: [main]" in ci
    assert "scripts/verify_clean_install.py" in ci
    assert "scripts/verify_source_archive.py" in ci
    assert "actions/checkout@v7" in ci and "actions/setup-python@v6" in ci
    assert "contents: write" in release
    assert "needs: release-gate" in release
    assert "scripts/fuzz_ghostline_levels.py --seeds 10000" in release
    assert "scripts/verify_release_evidence.py" in release
    assert "scripts/verify_security_release_evidence.py" in release
    assert "models/ghostline-security.pt" in release
    assert "benchmarks/neural/champion-final-8m-500.json" in release
    assert "champion-final-7m-500" not in release
    assert "benchmarks/security/**" in release
    assert "scripts/verify_source_archive.py" in release
    assert "scripts/verify_source_archive.py --release" in release
    assert "--release-smoke-test" in release
    assert "dist/Ghostline.manifest.json" in release
    assert "scripts/build_web.py --human-only" in ci
    assert "scripts/build_web.py --model models/ghostline-policy.onnx" in release
    assert "dist/LICENSE" in release
    assert "dist/ASSET-LICENSES.json" in release
    assert "dist/THIRD_PARTY_NOTICES.md" in release
    assert "dist/Ghostline.policy-parity.json" in release
    assert "dist/licenses/**" in release
    assert "node --test web/tests/*.test.mjs" in release
    assert "node --test web/tests/*.test.mjs" in ci
    assert "actions/download-artifact@v7" in release
    assert "Ghostline-Portfolio-Evidence" in release
    assert "videos/ghostline-demo.mp4" in release
    assert "benchmarks/neural/champion-onnx-parity.json" in release
    assert "benchmarks/system/headless-throughput.json" in release
    assert "gh release create" in release
    assert "--verify-tag" in release
