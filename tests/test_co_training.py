from __future__ import annotations

from pathlib import Path
import json

import pytest

import ghostline.co_training as co_training
from ghostline.co_training import (
    CoTrainingConfig,
    _cpu_affinity_mask,
    _training_process_environment,
    build_generation_plan,
    run_co_training,
)


def _config(tmp_path: Path) -> CoTrainingConfig:
    published = (
        Path(__file__).resolve().parents[1]
        / "models"
        / "ghostline-policy.pt"
    )
    return CoTrainingConfig(
        output=tmp_path / "league",
        published_runner=published,
        hours=0.01,
        generations=2,
        runner_envs=2,
        security_envs=1,
        runner_rollout=8,
        security_rollout=8,
        monitor_seconds=1.0,
    )


def test_generation_uses_only_previously_frozen_opponents(tmp_path: Path) -> None:
    config = _config(tmp_path)
    prior_runner = tmp_path / "runner-best.pt"
    prior_security = tmp_path / "security-best.pt"
    prior_runner.write_bytes(b"runner")
    prior_security.write_bytes(b"security")
    plan = build_generation_plan(
        config,
        generation=1,
        runner_pool=(prior_runner,),
        security_pool=(prior_security,),
        previous_runner=prior_runner,
        previous_security=prior_security,
    )

    assert plan.runner_opponents == (prior_security,)
    assert plan.security_opponents == (prior_runner,)
    assert "--security-opponent" in plan.runner_command
    assert "--runner-pool" in plan.security_command
    assert "--init-checkpoint" in plan.runner_command
    assert "--init-model" in plan.security_command
    assert str(plan.runner_output) not in plan.security_command
    assert str(plan.security_output) not in plan.runner_command


def test_generations_use_disjoint_training_and_validation_offsets(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    first = build_generation_plan(
        config,
        generation=0,
        runner_pool=(),
        security_pool=(),
        previous_runner=None,
        previous_security=None,
    )
    second_checkpoint = tmp_path / "selected.pt"
    second_checkpoint.write_bytes(b"selected")
    second = build_generation_plan(
        config,
        generation=1,
        runner_pool=(second_checkpoint,),
        security_pool=(second_checkpoint,),
        previous_runner=second_checkpoint,
        previous_security=second_checkpoint,
    )

    assert "0" in first.runner_command
    assert "200000" in second.runner_command
    assert "10000000" in first.security_command
    assert "10200000" in second.security_command
    assert first.runner_command != second.runner_command


def test_co_training_rejects_more_generations_than_seed_partition(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with pytest.raises(ValueError, match="generations"):
        CoTrainingConfig(
            **{
                **config.__dict__,
                "generations": 5,
            }
        ).validate()


def test_co_training_caps_implicit_numerical_thread_pools(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    environment = _training_process_environment(config)
    assert {
        environment[name]
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "BLIS_NUM_THREADS",
        )
    } == {"1"}
    with pytest.raises(ValueError, match="cpu_thread_limit"):
        CoTrainingConfig(
            **{
                **config.__dict__,
                "cpu_thread_limit": 0,
            }
        ).validate()


def test_co_training_affinity_is_a_hard_fractional_cpu_ceiling(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    mask, selected = _cpu_affinity_mask(24, config.cpu_fraction_limit)
    assert selected == 12
    assert mask.bit_count() == selected
    assert mask == (1 << selected) - 1
    with pytest.raises(ValueError, match="cpu_fraction_limit"):
        CoTrainingConfig(
            **{
                **config.__dict__,
                "cpu_fraction_limit": 0.75,
            }
        ).validate()


class _FinishedProcess:
    next_pid = 20_000

    def __init__(
        self,
        command: list[str] | tuple[str, ...],
        *,
        return_code: int = 0,
        **_: object,
    ) -> None:
        self.command = tuple(command)
        self.return_code = int(return_code)
        self.pid = _FinishedProcess.next_pid
        _FinishedProcess.next_pid += 1
        output = Path(self.command[self.command.index("--output") + 1])
        output.mkdir(parents=True, exist_ok=True)
        if "--dry-run" in self.command:
            config = {"tiers": [1, 2, 3, 4, 5, 6]}
            manifest = (
                {"status": "preflight-passed", "checkpoint_contract": {"config": config}}
                if "train-runner-v2" in self.command
                else {"status": "preflight-passed"}
            )
            (output / "experiment-manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
        elif "train-runner-v2" in self.command:
            (output / "latest.pt").write_bytes(b"runner-latest")
            (output / "best.pt").write_bytes(b"runner-best")
        else:
            (output / "latest.pt").write_bytes(b"security-latest")
            (output / "champion.pt").write_bytes(b"security-best")

    def poll(self) -> int:
        return self.return_code

    def terminate(self) -> None:
        self.return_code = -15

    def kill(self) -> None:
        self.return_code = -9

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        return self.return_code


def _disable_affinity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        co_training,
        "_apply_process_cpu_affinity",
        lambda _config: {
            "status": "test",
            "logical_cpu_count": 1,
            "selected_cpu_count": 1,
            "fraction_limit": 0.5,
            "mask": 1,
        },
    )


def test_driver_selects_and_freezes_each_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_affinity(monkeypatch)
    commands: list[tuple[str, ...]] = []

    def launch(command: list[str] | tuple[str, ...], **kwargs: object) -> _FinishedProcess:
        commands.append(tuple(command))
        return _FinishedProcess(command, **kwargs)

    monkeypatch.setattr(co_training.subprocess, "Popen", launch)
    config = _config(tmp_path)
    result_path = run_co_training(config)
    result = json.loads(result_path.read_text(encoding="utf-8"))

    assert result["status"] == "trained-candidates"
    assert len(result["generations"]) == 2
    assert all(
        record["status"] == "validation-selected"
        for record in result["generations"]
    )
    second_runner = next(
        command
        for command in commands
        if "generation-01" in " ".join(command)
        and "train-runner-v2" in command
    )
    second_security = next(
        command
        for command in commands
        if "generation-01" in " ".join(command)
        and "train-security" in command
    )
    assert "--security-opponent" in second_runner
    assert "--runner-pool" in second_security


def test_driver_resume_reuses_frozen_boundary_and_strict_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_affinity(monkeypatch)

    def fail_second_generation(
        command: list[str] | tuple[str, ...],
        **kwargs: object,
    ) -> _FinishedProcess:
        generation_one = "generation-01" in " ".join(command)
        return _FinishedProcess(
            command,
            return_code=7 if generation_one and "train-runner-v2" in command else 0,
            **kwargs,
        )

    monkeypatch.setattr(co_training.subprocess, "Popen", fail_second_generation)
    config = _config(tmp_path)
    with pytest.raises(RuntimeError, match="failed with return codes"):
        run_co_training(config)
    state = json.loads(
        (config.output / "session.json").read_text(encoding="utf-8")
    )
    assert state["generations"][0]["status"] == "validation-selected"
    assert state["generations"][1]["status"] == "interrupted"

    resumed_commands: list[tuple[str, ...]] = []

    def finish_resume(
        command: list[str] | tuple[str, ...],
        **kwargs: object,
    ) -> _FinishedProcess:
        resumed_commands.append(tuple(command))
        return _FinishedProcess(command, **kwargs)

    monkeypatch.setattr(co_training.subprocess, "Popen", finish_resume)
    resumed = CoTrainingConfig(**{**config.__dict__, "resume": True})
    result = json.loads(
        run_co_training(resumed).read_text(encoding="utf-8")
    )
    assert result["status"] == "trained-candidates"
    assert len(result["generations"]) == 2
    runner_command = next(
        command
        for command in resumed_commands
        if "train-runner-v2" in command
    )
    security_command = next(
        command
        for command in resumed_commands
        if "train-security" in command
    )
    assert "--resume" in runner_command
    assert "--published-v1-init" not in runner_command
    assert "--init-checkpoint" not in runner_command
    assert "--init-model" not in security_command


def test_driver_failure_terminates_the_concurrent_sibling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_affinity(monkeypatch)
    sibling: _PendingProcess | None = None

    class _PendingProcess(_FinishedProcess):
        def __init__(
            self,
            command: list[str] | tuple[str, ...],
            **kwargs: object,
        ) -> None:
            super().__init__(command, **kwargs)
            self.return_code: int | None = None
            self.was_terminated = False

        def poll(self) -> int | None:
            return self.return_code

        def terminate(self) -> None:
            self.was_terminated = True
            self.return_code = -15

    def launch(
        command: list[str] | tuple[str, ...],
        **kwargs: object,
    ) -> _FinishedProcess:
        nonlocal sibling
        if "train-runner-v2" in command:
            return _FinishedProcess(command, return_code=9, **kwargs)
        sibling = _PendingProcess(command, **kwargs)
        return sibling

    monkeypatch.setattr(co_training.subprocess, "Popen", launch)
    config = CoTrainingConfig(
        **{**_config(tmp_path).__dict__, "generations": 1}
    )
    with pytest.raises(RuntimeError, match="failed with return codes"):
        run_co_training(config)
    assert sibling is not None and sibling.was_terminated


def test_driver_dry_run_executes_generation_boundary_preflight(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _disable_affinity(monkeypatch)
    monkeypatch.setattr(co_training.subprocess, "Popen", _FinishedProcess)
    base = _config(tmp_path)
    config = CoTrainingConfig(
        **{**base.__dict__, "generations": 1, "dry_run": True}
    )

    result = json.loads(
        run_co_training(config).read_text(encoding="utf-8")
    )
    assert result["status"] == "preflight-passed"
    assert result["generations"][0]["status"] == "preflight-passed"
