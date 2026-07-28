"""Generation-based concurrent league training for Ghostline v2.

Runner PPO and security MAPPO improve during the same wall-clock session, but
each generation sees only immutable opponents selected by held-out validation
in earlier generations.  This keeps both updates on-policy and removes the
moving-target instability of naive live self-play.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Sequence


CO_TRAINING_CONTRACT = "ghostline-v2-frozen-league-cotraining-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _tail_jsonl(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    last = ""
    with path.open("r", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                last = line
    if not last:
        return None
    try:
        value = json.loads(last)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


@dataclass(frozen=True)
class CoTrainingConfig:
    output: Path
    published_runner: Path
    hours: float = 24.0
    generations: int = 3
    runner_envs: int = 16
    security_envs: int = 8
    runner_rollout: int = 512
    security_rollout: int = 192
    runner_epochs: int = 4
    security_epochs: int = 2
    recurrent_size_runner: int = 384
    recurrent_size_security: int = 256
    runner_learning_rate: float = 5.0e-5
    runner_entropy_coefficient: float = 0.003
    runner_initial_curriculum_tier: int = 3
    runner_ghost_directive_fraction: float = 0.50
    security_learning_rate: float = 3.0e-4
    gamma: float = 0.999
    gae_lambda: float = 0.98
    reward_scale: float = 0.05
    monitor_seconds: float = 30.0
    runner_validation_interval: int = 100
    runner_validation_episodes: int = 10
    security_validation_interval: int = 500_000
    security_validation_episodes: int = 10
    security_bc_steps: int = 50_000
    scripted_opponent_fraction: float = 0.20
    runner_max_decisions: int = 0
    security_max_steps: int = 0
    cpu_thread_limit: int = 1
    cpu_fraction_limit: float = 0.50
    resume: bool = False
    dry_run: bool = False

    def validate(self) -> None:
        if not self.published_runner.is_file():
            raise FileNotFoundError(
                f"published runner checkpoint is missing: {self.published_runner}"
            )
        if self.hours <= 0.0:
            raise ValueError("hours must be positive")
        if not 1 <= self.generations <= 4:
            raise ValueError("generations must lie in 1..4")
        if self.runner_envs < 1 or self.security_envs < 1:
            raise ValueError("runner_envs and security_envs must be positive")
        if self.runner_rollout < 2 or self.security_rollout < 2:
            raise ValueError("rollouts must contain at least two decisions")
        if self.runner_epochs < 1 or self.security_epochs < 1:
            raise ValueError("training epochs must be positive")
        if self.recurrent_size_runner != 384:
            raise ValueError(
                "published-v1 league initialization requires runner recurrent size 384"
            )
        if self.recurrent_size_security not in (256, 384):
            raise ValueError("security recurrent size must be 256 or 384")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must lie in (0, 1]")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must lie in [0, 1]")
        if not 0.0 < self.reward_scale <= 1.0:
            raise ValueError("reward_scale must lie in (0, 1]")
        if self.runner_learning_rate <= 0.0 or self.security_learning_rate <= 0.0:
            raise ValueError("learning rates must be positive")
        if not 0.0 <= self.runner_entropy_coefficient <= 0.10:
            raise ValueError("runner entropy coefficient must lie in [0, 0.10]")
        if self.runner_initial_curriculum_tier not in range(1, 7):
            raise ValueError("runner initial curriculum tier must lie in 1..6")
        if not 0.0 <= self.runner_ghost_directive_fraction <= 1.0:
            raise ValueError(
                "runner ghost directive fraction must lie in [0, 1]"
            )
        if not 1.0 <= self.monitor_seconds <= 60.0:
            raise ValueError("monitor_seconds must lie in 1..60")
        if not 0.0 <= self.scripted_opponent_fraction <= 1.0:
            raise ValueError("scripted_opponent_fraction must lie in [0, 1]")
        if self.runner_max_decisions < 0 or self.security_max_steps < 0:
            raise ValueError("step limits cannot be negative")
        if not 1 <= self.cpu_thread_limit <= 4:
            raise ValueError("cpu_thread_limit must lie in 1..4")
        if not 0.10 <= self.cpu_fraction_limit <= 0.60:
            raise ValueError("cpu_fraction_limit must lie in [0.10, 0.60]")


@dataclass(frozen=True)
class GenerationPlan:
    index: int
    runner_output: Path
    security_output: Path
    runner_command: tuple[str, ...]
    security_command: tuple[str, ...]
    runner_opponents: tuple[Path, ...]
    security_opponents: tuple[Path, ...]


def build_generation_plan(
    config: CoTrainingConfig,
    *,
    generation: int,
    runner_pool: Sequence[Path],
    security_pool: Sequence[Path],
    previous_runner: Path | None,
    previous_security: Path | None,
    resume_runner: bool = False,
    resume_security: bool = False,
) -> GenerationPlan:
    """Build one deterministic pair of frozen-opponent training commands."""

    config.validate()
    if not 0 <= generation < config.generations:
        raise ValueError("generation index leaves the configured campaign")
    phase_hours = config.hours / config.generations
    generation_dir = config.output / f"generation-{generation:02d}"
    runner_output = generation_dir / "runner"
    security_output = generation_dir / "security"
    runner_training_start = generation * 200_000
    security_training_start = 10_000_000 + generation * 200_000
    # The runner contract reserves 8k held-out episodes per tier. Four league
    # generations therefore receive disjoint 2k windows. At the calibrated
    # validation cadence a 16-hour generation uses well under half its window.
    validation_cursor = generation * 2_000

    runner: list[str] = [
        sys.executable,
        "-m",
        "ghostline",
        "train-runner-v2",
        "--output",
        str(runner_output),
        "--envs",
        str(config.runner_envs),
        "--rollout",
        str(config.runner_rollout),
        "--epochs",
        str(config.runner_epochs),
        "--minibatch-envs",
        str(min(4, config.runner_envs)),
        "--recurrent-size",
        str(config.recurrent_size_runner),
        "--learning-rate",
        str(config.runner_learning_rate),
        "--entropy-coefficient",
        str(config.runner_entropy_coefficient),
        "--gamma",
        str(config.gamma),
        "--gae-lambda",
        str(config.gae_lambda),
        "--reward-scale",
        str(config.reward_scale),
        "--training-seed-start",
        str(runner_training_start),
        "--initial-validation-cursor",
        str(validation_cursor),
        "--initial-curriculum-tier",
        str(config.runner_initial_curriculum_tier),
        "--ghost-directive-fraction",
        str(config.runner_ghost_directive_fraction),
        "--validation-interval",
        str(config.runner_validation_interval),
        "--validation-episodes",
        str(config.runner_validation_episodes),
        "--seconds",
        str(phase_hours * 3600.0),
    ]
    if config.runner_max_decisions:
        runner.extend(
            ("--max-decisions", str(config.runner_max_decisions))
        )
    if resume_runner:
        runner.append("--resume")
    elif previous_runner is None:
        runner.extend(
            ("--published-v1-init", str(config.published_runner))
        )
    else:
        runner.extend(("--init-checkpoint", str(previous_runner)))
    for checkpoint in security_pool:
        runner.extend(("--security-opponent", str(checkpoint)))

    security: list[str] = [
        sys.executable,
        "-m",
        "ghostline",
        "train-security",
        "--output",
        str(security_output),
        "--hours",
        str(phase_hours),
        "--envs",
        str(config.security_envs),
        "--rollout",
        str(config.security_rollout),
        "--epochs",
        str(config.security_epochs),
        "--recurrent-size",
        str(config.recurrent_size_security),
        "--learning-rate",
        str(config.security_learning_rate),
        "--gamma",
        str(config.gamma),
        "--gae-lambda",
        str(config.gae_lambda),
        "--reward-scale",
        str(config.reward_scale),
        "--training-seed-start",
        str(security_training_start),
        "--initial-validation-cursor",
        str(validation_cursor),
        "--validation-interval",
        str(config.security_validation_interval),
        "--validation-episodes",
        str(config.security_validation_episodes),
        "--runner-model",
        str(config.published_runner),
        "--scripted-opponent-fraction",
        str(config.scripted_opponent_fraction),
    ]
    if config.security_max_steps:
        security.extend(("--max-steps", str(config.security_max_steps)))
    if resume_security:
        security.extend(
            (
                "--bc-warmup-steps",
                str(
                    config.security_bc_steps
                    if previous_security is None
                    else 0
                ),
            )
        )
    elif previous_security is None:
        security.extend(
            ("--bc-warmup-steps", str(config.security_bc_steps))
        )
    else:
        security.extend(
            (
                "--init-model",
                str(previous_security),
                "--bc-warmup-steps",
                "0",
            )
        )
    for checkpoint in runner_pool:
        security.extend(("--runner-pool", str(checkpoint)))

    if config.dry_run:
        runner.append("--dry-run")
        security.append("--dry-run")
    return GenerationPlan(
        index=generation,
        runner_output=runner_output,
        security_output=security_output,
        runner_command=tuple(runner),
        security_command=tuple(security),
        runner_opponents=tuple(Path(path) for path in security_pool),
        security_opponents=tuple(Path(path) for path in runner_pool),
    )


def _selected_checkpoint(output: Path, side: str) -> Path:
    selected = (
        output / "best.pt"
        if side == "runner"
        else output / "champion.pt"
    )
    if not selected.is_file():
        raise RuntimeError(
            f"{side} generation completed without a validation-selected "
            f"checkpoint: {selected}"
        )
    return selected.resolve()


def _terminate(processes: Iterable[subprocess.Popen[str]]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    for process in processes:
        if process.poll() is None:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()


def _training_process_environment(
    config: CoTrainingConfig,
) -> dict[str, str]:
    """Bound implicit numerical-library pools inside every trainer process."""

    environment = os.environ.copy()
    thread_limit = str(config.cpu_thread_limit)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        environment[name] = thread_limit
    return environment


def _cpu_affinity_mask(
    logical_cpu_count: int,
    fraction: float,
) -> tuple[int, int]:
    available = max(1, int(logical_cpu_count))
    selected = max(1, min(available, math.floor(available * fraction)))
    return (1 << selected) - 1, selected


def _apply_process_cpu_affinity(
    config: CoTrainingConfig,
) -> dict[str, int | float | str]:
    logical = max(1, int(os.cpu_count() or 1))
    mask, selected = _cpu_affinity_mask(
        logical,
        config.cpu_fraction_limit,
    )
    if os.name == "nt":
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetProcessAffinityMask.argtypes = (
            wintypes.HANDLE,
            ctypes.c_size_t,
        )
        kernel32.SetProcessAffinityMask.restype = wintypes.BOOL
        handle = kernel32.GetCurrentProcess()
        if not kernel32.SetProcessAffinityMask(handle, mask):
            raise OSError(
                ctypes.get_last_error(),
                "failed to apply the co-training CPU affinity ceiling",
            )
        status = "applied"
    else:
        status = "unsupported-platform"
    return {
        "status": status,
        "logical_cpu_count": logical,
        "selected_cpu_count": selected,
        "fraction_limit": config.cpu_fraction_limit,
        "mask": mask,
    }


def _campaign_configuration(config: CoTrainingConfig) -> dict[str, Any]:
    """Return the immutable campaign contract used to authorize a resume."""

    payload = asdict(config)
    payload.pop("resume", None)
    payload.pop("dry_run", None)
    payload["output"] = str(config.output.resolve())
    payload["published_runner"] = str(config.published_runner.resolve())
    return payload


def _session_payload(
    config: CoTrainingConfig,
    *,
    created: str,
    status: str,
    records: Sequence[dict[str, Any]],
    started: float,
    affinity: dict[str, int | float | str],
    error: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "contract": CO_TRAINING_CONTRACT,
        "created_at_utc": created,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "elapsed_seconds": time.monotonic() - started,
        "campaign_configuration": _campaign_configuration(config),
        "resource_controls": {
            "cpu_thread_limit": config.cpu_thread_limit,
            "cpu_affinity": affinity,
            "windows_priority": (
                "below-normal" if os.name == "nt" else "platform-default"
            ),
        },
        "generations": list(records),
    }
    if error is not None:
        payload["error"] = error
    return payload


def _load_completed_generations(
    config: CoTrainingConfig,
    state_path: Path,
) -> tuple[str, list[dict[str, Any]], list[Path], list[Path]]:
    """Validate a prior session and recover only frozen generation boundaries."""

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot resume invalid session state: {state_path}") from exc
    if state.get("contract") != CO_TRAINING_CONTRACT:
        raise RuntimeError("cannot resume a different co-training contract")
    expected = _campaign_configuration(config)
    if state.get("campaign_configuration") != expected:
        raise RuntimeError(
            "cannot resume because the co-training configuration changed"
        )
    completed: list[dict[str, Any]] = []
    runner_pool: list[Path] = []
    security_pool: list[Path] = []
    for expected_index, record in enumerate(state.get("generations", [])):
        if record.get("status") != "validation-selected":
            break
        if int(record.get("generation", -1)) != expected_index:
            raise RuntimeError("resume state has a non-contiguous generation history")
        runner = Path(record["runner_checkpoint"]["path"]).resolve()
        security = Path(record["security_checkpoint"]["path"]).resolve()
        if not runner.is_file() or _sha256(runner) != record["runner_checkpoint"]["sha256"]:
            raise RuntimeError(f"frozen runner checkpoint changed or disappeared: {runner}")
        if (
            not security.is_file()
            or _sha256(security) != record["security_checkpoint"]["sha256"]
        ):
            raise RuntimeError(
                f"frozen security checkpoint changed or disappeared: {security}"
            )
        completed.append(record)
        runner_pool.append(runner)
        security_pool.append(security)
    return (
        str(state.get("created_at_utc") or datetime.now(timezone.utc).isoformat()),
        completed,
        runner_pool,
        security_pool,
    )


def _validate_dry_run_outputs(plan: GenerationPlan) -> None:
    """Require both trainer preflights to execute before a driver preflight passes."""

    for side, output in (
        ("runner", plan.runner_output),
        ("security", plan.security_output),
    ):
        manifest_path = output / "experiment-manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"{side} dry-run did not produce a valid experiment manifest"
            ) from exc
        if manifest.get("status") != "preflight-passed":
            raise RuntimeError(f"{side} dry-run preflight did not pass")
    runner_manifest = json.loads(
        (plan.runner_output / "experiment-manifest.json").read_text(
            encoding="utf-8"
        )
    )
    configured_tiers = tuple(
        int(tier)
        for tier in runner_manifest["checkpoint_contract"]["config"]["tiers"]
    )
    if configured_tiers != (1, 2, 3, 4, 5, 6):
        raise RuntimeError(
            "runner selection preflight must cover every contract tier"
        )


def run_co_training(config: CoTrainingConfig) -> Path:
    """Run the frozen-snapshot league and return its result manifest."""

    config.validate()
    affinity = _apply_process_cpu_affinity(config)
    config.output.mkdir(parents=True, exist_ok=True)
    state_path = config.output / "session.json"
    result_path = config.output / "result.json"
    if config.resume and result_path.is_file():
        return result_path
    if not config.resume and (state_path.exists() or result_path.exists()):
        raise RuntimeError(
            f"{config.output} already contains a co-training session; "
            "choose a new output directory or pass --resume"
        )
    started = time.monotonic()
    if config.resume:
        if not state_path.is_file():
            raise RuntimeError(
                f"cannot resume because {state_path} does not exist"
            )
        created, records, runner_pool, security_pool = (
            _load_completed_generations(config, state_path)
        )
    else:
        created = datetime.now(timezone.utc).isoformat()
        records = []
        runner_pool = []
        security_pool = []
    previous_runner = runner_pool[-1] if runner_pool else None
    previous_security = security_pool[-1] if security_pool else None

    for generation in range(len(records), config.generations):
        generation_dir = config.output / f"generation-{generation:02d}"
        resume_runner = (
            config.resume
            and (generation_dir / "runner" / "latest.pt").is_file()
        )
        resume_security = (
            config.resume
            and (generation_dir / "security" / "latest.pt").is_file()
        )
        plan = build_generation_plan(
            config,
            generation=generation,
            runner_pool=runner_pool,
            security_pool=security_pool,
            previous_runner=previous_runner,
            previous_security=previous_security,
            resume_runner=resume_runner,
            resume_security=resume_security,
        )
        plan.runner_output.mkdir(parents=True, exist_ok=True)
        plan.security_output.mkdir(parents=True, exist_ok=True)
        runner_log = plan.runner_output / "process.log"
        security_log = plan.security_output / "process.log"
        generation_record: dict[str, Any] = {
            "generation": generation,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "runner_command": list(plan.runner_command),
            "security_command": list(plan.security_command),
            "runner_opponents": [
                {"path": str(path), "sha256": _sha256(path)}
                for path in plan.runner_opponents
            ],
            "security_opponents": [
                {"path": str(path), "sha256": _sha256(path)}
                for path in plan.security_opponents
            ],
            "resource_controls": {
                "cpu_thread_limit": config.cpu_thread_limit,
                "cpu_affinity": affinity,
                "windows_priority": (
                    "below-normal"
                    if os.name == "nt"
                    else "platform-default"
                ),
            },
            "status": "running",
            "resumed": bool(resume_runner or resume_security),
        }
        records.append(generation_record)
        _atomic_json(
            state_path,
            _session_payload(
                config,
                created=created,
                status="running",
                records=records,
                started=started,
                affinity=affinity,
            ),
        )

        stream_mode = "a" if resume_runner or resume_security else "w"
        streams = (
            runner_log.open(stream_mode, encoding="utf-8", newline="\n"),
            security_log.open(stream_mode, encoding="utf-8", newline="\n"),
        )
        processes: list[subprocess.Popen[str]] = []
        try:
            for command, stream in zip(
                (plan.runner_command, plan.security_command),
                streams,
                strict=True,
            ):
                processes.append(
                    subprocess.Popen(
                        command,
                        cwd=Path.cwd(),
                        env=_training_process_environment(config),
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                        text=True,
                        creationflags=(
                            getattr(
                                subprocess,
                                "BELOW_NORMAL_PRIORITY_CLASS",
                                0,
                            )
                            if os.name == "nt"
                            else 0
                        ),
                    )
                )
            generation_record["pids"] = [item.pid for item in processes]
            while True:
                return_codes = [item.poll() for item in processes]
                generation_record["runner_metrics"] = _tail_jsonl(
                    plan.runner_output / "metrics.jsonl"
                )
                generation_record["security_metrics"] = _tail_jsonl(
                    plan.security_output / "training-metrics.jsonl"
                )
                _atomic_json(
                    state_path,
                    _session_payload(
                        config,
                        created=created,
                        status="running",
                        records=records,
                        started=started,
                        affinity=affinity,
                    ),
                )
                failures = [
                    code for code in return_codes
                    if code is not None and code != 0
                ]
                if failures:
                    _terminate(processes)
                    raise RuntimeError(
                        f"co-training generation {generation} failed with "
                        f"return codes {return_codes}; inspect {runner_log} and "
                        f"{security_log}"
                    )
                if all(code == 0 for code in return_codes):
                    break
                time.sleep(config.monitor_seconds)
        except Exception as exc:
            generation_record["status"] = "interrupted"
            generation_record["error"] = str(exc)
            _atomic_json(
                state_path,
                _session_payload(
                    config,
                    created=created,
                    status="interrupted",
                    records=records,
                    started=started,
                    affinity=affinity,
                    error=str(exc),
                ),
            )
            raise
        finally:
            _terminate(processes)
            for stream in streams:
                stream.close()

        if config.dry_run:
            _validate_dry_run_outputs(plan)
            generation_record["status"] = "preflight-passed"
            continue
        try:
            previous_runner = _selected_checkpoint(
                plan.runner_output,
                "runner",
            )
            previous_security = _selected_checkpoint(
                plan.security_output,
                "security",
            )
        except Exception as exc:
            generation_record["status"] = "interrupted"
            generation_record["error"] = str(exc)
            _atomic_json(
                state_path,
                _session_payload(
                    config,
                    created=created,
                    status="interrupted",
                    records=records,
                    started=started,
                    affinity=affinity,
                    error=str(exc),
                ),
            )
            raise
        runner_pool.append(previous_runner)
        security_pool.append(previous_security)
        generation_record.update(
            {
                "status": "validation-selected",
                "finished_at_utc": datetime.now(timezone.utc).isoformat(),
                "runner_checkpoint": {
                    "path": str(previous_runner),
                    "sha256": _sha256(previous_runner),
                },
                "security_checkpoint": {
                    "path": str(previous_security),
                    "sha256": _sha256(previous_security),
                },
            }
        )
        _atomic_json(
            state_path,
            _session_payload(
                config,
                created=created,
                status="running",
                records=records,
                started=started,
                affinity=affinity,
            ),
        )

    result: dict[str, Any] = {
        "contract": CO_TRAINING_CONTRACT,
        "created_at_utc": created,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - started,
        "status": "preflight-passed" if config.dry_run else "trained-candidates",
        "generations": records,
    }
    if not config.dry_run:
        assert previous_runner is not None and previous_security is not None
        final_runner = config.output / "runner-v2-candidate.pt"
        final_security = config.output / "security-v2-candidate.pt"
        shutil.copy2(previous_runner, final_runner)
        shutil.copy2(previous_security, final_security)
        result["runner_candidate"] = {
            "path": str(final_runner.resolve()),
            "sha256": _sha256(final_runner),
        }
        result["security_candidate"] = {
            "path": str(final_security.resolve()),
            "sha256": _sha256(final_security),
        }
    _atomic_json(result_path, result)
    final_state = _session_payload(
        config,
        created=created,
        status=result["status"],
        records=records,
        started=started,
        affinity=affinity,
    )
    final_state["result"] = str(result_path.resolve())
    _atomic_json(
        state_path,
        final_state,
    )
    return result_path
