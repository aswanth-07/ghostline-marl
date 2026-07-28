from __future__ import annotations

from pathlib import Path

import pytest

from ghostline.co_training import (
    CoTrainingConfig,
    _cpu_affinity_mask,
    _training_process_environment,
    build_generation_plan,
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
