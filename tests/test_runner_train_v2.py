from __future__ import annotations

from copy import deepcopy
from argparse import Namespace
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from ghostline.curriculum import ACCEPTANCE_THRESHOLDS
from ghostline.model_v2 import RunnerPolicyV2
from ghostline.runner_train_v2 import (
    ALL_TIERS,
    EXPERIMENT_MANIFEST_CONTRACT,
    OBSERVATION_KEYS_V2,
    RunnerPPOConfig,
    ScheduledRunnerEnv,
    build_parser,
    _require_checkpoint_contract,
    acceptance_gate,
    collect_rollout,
    compute_gae,
    curriculum_gate,
    evaluate_rollout_log_probabilities,
    initialize_fresh_policy,
    load_training_checkpoint,
    make_runner_vector_env,
    observation_digest,
    ppo_update,
    public_auxiliary_labels,
    require_training_schedule,
    require_validation_window,
    save_training_checkpoint,
    selection_validation_tiers,
    runner_sequence_outputs,
    validate_runner,
    validate_runner_suite,
    validation_selection_key,
    train,
)
from ghostline.seeds import TRAINING_SEED_END, validation_seed
from ghostline.types_v2 import (
    RUNNER_ACTION_COUNT_V2,
    ContractDirective,
)


def smoke_config(**changes: object) -> RunnerPPOConfig:
    values: dict[str, object] = {
        "seed": 19,
        "envs": 1,
        "rollout": 2,
        "epochs": 1,
        "minibatch_envs": 1,
        "recurrent_size": 256,
        "async_envs": False,
        "validation_interval": 0,
        "validation_episodes": 0,
    }
    values.update(changes)
    return RunnerPPOConfig(**values)


def test_vector_rollout_uses_complete_288_action_mask_and_exact_log_probs() -> None:
    config = smoke_config()
    envs = make_runner_vector_env(config)
    try:
        observation, _ = envs.reset()
        assert tuple(sorted(observation)) == tuple(sorted(OBSERVATION_KEYS_V2))
        assert observation["local_grid"].shape == (1, 15, 15, 15)
        assert observation["field_targets"].shape[-2:] == (16, 13)
        assert observation["action_mask"].shape == (1, RUNNER_ACTION_COUNT_V2)

        torch.manual_seed(123)
        policy = RunnerPolicyV2(recurrent_size=256)
        rollout = collect_rollout(
            policy=policy,
            vector_env=envs,
            observation=observation,
            hidden=torch.zeros(1, 1, 256),
            episode_starts=np.ones(1, dtype=bool),
            rollout_steps=config.rollout,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            device=torch.device("cpu"),
        )
        sampled_legality = np.take_along_axis(
            rollout.observations["action_mask"],
            rollout.actions[..., None],
            axis=-1,
        ).squeeze(-1)
        assert np.all(sampled_legality == 1)
        assert np.all((0 <= rollout.actions) & (rollout.actions < 288))

        log_probability, values, _ = evaluate_rollout_log_probabilities(
            policy,
            rollout,
            device=torch.device("cpu"),
        )
        assert torch.equal(
            log_probability,
            torch.as_tensor(rollout.old_log_probabilities),
        )
        assert torch.allclose(
            values,
            torch.as_tensor(rollout.old_values),
            atol=1e-6,
            rtol=1e-6,
        )
    finally:
        envs.close()


def test_gae_matches_hand_calculation_and_cuts_terminal_bootstrap() -> None:
    rewards = np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32)
    values = np.asarray([[0.5], [0.75], [1.0]], dtype=np.float32)
    dones = np.asarray([[False], [True], [False]])
    advantages, returns = compute_gae(
        rewards,
        values,
        dones,
        np.asarray([4.0], dtype=np.float32),
        gamma=0.9,
        gae_lambda=0.8,
    )
    expected_last = 3.0 + 0.9 * 4.0 - 1.0
    expected_middle = 2.0 - 0.75
    expected_first = 1.0 + 0.9 * 0.75 - 0.5 + 0.9 * 0.8 * expected_middle
    assert advantages[:, 0] == pytest.approx(
        (expected_first, expected_middle, expected_last)
    )
    assert returns == pytest.approx(advantages + values)


def test_runner_sequence_reset_is_identical_to_stepwise_gru() -> None:
    config = smoke_config(rollout=3)
    envs = make_runner_vector_env(config)
    try:
        observation, _ = envs.reset()
        sequence = {
            key: torch.as_tensor(np.repeat(value[None], 3, axis=0))
            for key, value in observation.items()
        }
        resets = torch.tensor([[True], [False], [True]])
        torch.manual_seed(211)
        policy = RunnerPolicyV2(recurrent_size=256).eval()
        initial = torch.randn(1, 1, 256)
        sequence_logits, sequence_values, sequence_hidden = policy.forward_sequence(
            sequence,
            initial.clone(),
            resets,
        )

        hidden = initial.clone()
        step_logits = []
        step_values = []
        for index in range(3):
            if resets[index, 0]:
                hidden.zero_()
            current = {key: value[index] for key, value in sequence.items()}
            logits, values, hidden = policy(current, hidden)
            step_logits.append(logits)
            step_values.append(values)
        assert torch.allclose(sequence_logits, torch.stack(step_logits), atol=1e-6)
        assert torch.allclose(sequence_values, torch.stack(step_values), atol=1e-6)
        assert torch.allclose(sequence_hidden, hidden, atol=1e-6)
    finally:
        envs.close()


def test_ppo_smoke_has_finite_kl_clip_and_grad_diagnostics() -> None:
    config = smoke_config()
    envs = make_runner_vector_env(config)
    try:
        observation, _ = envs.reset()
        torch.manual_seed(317)
        policy = RunnerPolicyV2(recurrent_size=256)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=config.learning_rate)
        objective_head_before = policy.objective_head.weight.detach().clone()
        danger_head_before = policy.danger_head.weight.detach().clone()
        rollout = collect_rollout(
            policy=policy,
            vector_env=envs,
            observation=observation,
            hidden=torch.zeros(1, 1, 256),
            episode_starts=np.ones(1, dtype=bool),
            rollout_steps=config.rollout,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            device=torch.device("cpu"),
        )
        diagnostics = ppo_update(
            policy=policy,
            optimizer=optimizer,
            rollout=rollout,
            config=config,
            rng=np.random.default_rng(41),
            device=torch.device("cpu"),
        )
        assert diagnostics.samples == 2
        assert diagnostics.epochs_completed == 1
        assert all(
            np.isfinite(value)
            for value in (
                diagnostics.policy_loss,
                diagnostics.value_loss,
                diagnostics.entropy,
                diagnostics.objective_aux_loss,
                diagnostics.danger_aux_loss,
                diagnostics.weighted_auxiliary_loss,
                diagnostics.approximate_kl,
                diagnostics.clip_fraction,
                diagnostics.gradient_norm,
                diagnostics.explained_variance,
            )
        )
        assert diagnostics.approximate_kl >= -1e-6
        assert 0.0 <= diagnostics.clip_fraction <= 1.0
        assert diagnostics.gradient_norm >= 0.0
        assert diagnostics.objective_aux_loss >= 0.0
        assert diagnostics.danger_aux_loss >= 0.0
        assert diagnostics.weighted_auxiliary_loss >= 0.0
        assert not torch.equal(
            policy.objective_head.weight,
            objective_head_before,
        )
        assert not torch.equal(
            policy.danger_head.weight,
            danger_head_before,
        )
    finally:
        envs.close()


def test_nonfinite_rollout_aborts_before_optimizer_step() -> None:
    config = smoke_config()
    envs = make_runner_vector_env(config)
    try:
        observation, _ = envs.reset()
        policy = RunnerPolicyV2(recurrent_size=256)
        optimizer = torch.optim.AdamW(policy.parameters(), lr=config.learning_rate)
        rollout = collect_rollout(
            policy=policy,
            vector_env=envs,
            observation=observation,
            hidden=torch.zeros(1, 1, 256),
            episode_starts=np.ones(1, dtype=bool),
            rollout_steps=config.rollout,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            device=torch.device("cpu"),
        )
        rollout.returns[0, 0] = np.nan
        before = {
            name: value.detach().clone()
            for name, value in policy.state_dict().items()
        }
        with pytest.raises(FloatingPointError, match="optimizer step aborted"):
            ppo_update(
                policy=policy,
                optimizer=optimizer,
                rollout=rollout,
                config=config,
                rng=np.random.default_rng(53),
                device=torch.device("cpu"),
            )
        assert all(
            torch.equal(before[name], value)
            for name, value in policy.state_dict().items()
        )
    finally:
        envs.close()


def test_auxiliary_labels_are_public_and_share_one_recurrent_traversal() -> None:
    config = smoke_config(rollout=3)
    envs = make_runner_vector_env(config)
    try:
        observation, _ = envs.reset()
        sequence = {
            key: torch.as_tensor(np.repeat(value[None], 3, axis=0))
            for key, value in observation.items()
        }
        # Pin labels so their exact public origin is independently visible.
        sequence["objective"].zero_()
        sequence["objective"][0, 0, 1:3] = torch.tensor((3.0, 4.0))
        sequence["objective"][1, 0, 4:6] = torch.tensor((0.0, -2.0))
        sequence["rays"].zero_()
        sequence["rays"][0, 0, 7, 1] = 0.65
        sequence["rays"][1, 0, 3, 1] = 0.25
        bearing, danger = public_auxiliary_labels(sequence)
        assert bearing[0, 0] == pytest.approx(torch.tensor((0.6, 0.8)))
        assert bearing[1, 0] == pytest.approx(torch.tensor((0.0, -1.0)))
        assert bearing[2, 0] == pytest.approx(torch.zeros(2))
        assert danger[:, 0] == pytest.approx(torch.tensor((0.65, 0.25, 0.0)))

        policy = RunnerPolicyV2(recurrent_size=256)
        resets = torch.tensor([[True], [False], [True]])
        with patch.object(
            policy.core,
            "forward",
            wraps=policy.core.forward,
        ) as recurrent_forward:
            outputs = runner_sequence_outputs(
                policy,
                sequence,
                torch.zeros(1, 1, 256),
                resets,
            )
        # Reset-aware recurrence calls the GRU once per reset-delimited chunk.
        # A separate auxiliary traversal would double this count.
        assert recurrent_forward.call_count == 2
        assert outputs.logits.shape == (3, 1, RUNNER_ACTION_COUNT_V2)
        assert outputs.values.shape == (3, 1)
        assert outputs.objective_bearing.shape == (3, 1, 2)
        assert outputs.danger.shape == (3, 1)
    finally:
        envs.close()


def test_strict_checkpoint_restores_episode_rng_and_fails_on_contract_drift(
    tmp_path: Path,
) -> None:
    config = smoke_config(rollout=1)
    envs = make_runner_vector_env(config)
    torch.manual_seed(401)
    np.random.seed(401)
    rng = np.random.default_rng(401)
    policy = RunnerPolicyV2(recurrent_size=256)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=config.learning_rate)
    try:
        observation, _ = envs.reset()
        rollout = collect_rollout(
            policy=policy,
            vector_env=envs,
            observation=observation,
            hidden=torch.zeros(1, 1, 256),
            episode_starts=np.ones(1, dtype=bool),
            rollout_steps=1,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            device=torch.device("cpu"),
        )
        path = tmp_path / "latest.pt"
        save_training_checkpoint(
            path,
            policy=policy,
            optimizer=optimizer,
            config=config,
            rng=rng,
            vector_env=envs,
            observation=rollout.next_observation,
            hidden=rollout.next_hidden,
            episode_starts=rollout.next_episode_starts,
            updates=3,
            decisions=17,
            validation_cursor=20,
            acceptance_passes=1,
            curriculum_tier=1,
            promotion_passes=1,
            best_selection_key=(0.2, 0.3, -0.1, -20.0),
            next_validation_update=4,
            validation_history=[],
            initialization={"method": "test-scratch", "seed": 401},
        )
        expected_continuation = collect_rollout(
            policy=policy,
            vector_env=envs,
            observation=rollout.next_observation,
            hidden=rollout.next_hidden,
            episode_starts=rollout.next_episode_starts,
            rollout_steps=1,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            device=torch.device("cpu"),
        )
        expected_torch = torch.rand(3)
        expected_numpy_global = np.random.random(3)
        expected_numpy_local = rng.random(3)
        assert not list(tmp_path.glob("*.tmp"))
    finally:
        envs.close()

    restored_envs = make_runner_vector_env(config)
    restored_policy = RunnerPolicyV2(recurrent_size=256)
    restored_optimizer = torch.optim.AdamW(
        restored_policy.parameters(),
        lr=config.learning_rate,
    )
    restored_rng = np.random.default_rng(999)
    try:
        state = load_training_checkpoint(
            path,
            policy=restored_policy,
            optimizer=restored_optimizer,
            config=config,
            rng=restored_rng,
            vector_env=restored_envs,
            device=torch.device("cpu"),
        )
        assert state["updates"] == 3
        assert state["decisions"] == 17
        assert state["validation_cursor"] == 20
        assert state["acceptance_passes"] == 1
        assert state["curriculum_tier"] == 1
        assert state["promotion_passes"] == 1
        assert state["initialization"] == {
            "method": "test-scratch",
            "seed": 401,
        }
        assert observation_digest(state["observation"]) == observation_digest(
            rollout.next_observation
        )
        restored_continuation = collect_rollout(
            policy=restored_policy,
            vector_env=restored_envs,
            observation=state["observation"],
            hidden=state["hidden"],
            episode_starts=state["episode_starts"],
            rollout_steps=1,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            device=torch.device("cpu"),
        )
        assert np.array_equal(
            restored_continuation.actions,
            expected_continuation.actions,
        )
        assert restored_continuation.rewards == pytest.approx(
            expected_continuation.rewards
        )
        assert observation_digest(
            restored_continuation.next_observation
        ) == observation_digest(expected_continuation.next_observation)
        assert torch.equal(torch.rand(3), expected_torch)
        assert np.random.random(3) == pytest.approx(expected_numpy_global)
        assert restored_rng.random(3) == pytest.approx(expected_numpy_local)
    finally:
        restored_envs.close()

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    changed = deepcopy(checkpoint)
    changed["contract"]["action_count"] = 72
    with pytest.raises(RuntimeError, match="resume contract changed"):
        _require_checkpoint_contract(changed, config)


def test_published_v1_warm_start_is_mutually_exclusive_and_provenanced() -> None:
    checkpoint = (
        Path(__file__).resolve().parents[1]
        / "models"
        / "ghostline-policy.pt"
    )
    config = smoke_config(recurrent_size=384)
    args = Namespace(
        resume=False,
        init_checkpoint=None,
        published_v1_init=checkpoint,
    )
    policy, metadata = initialize_fresh_policy(
        args,
        config,
        torch.device("cpu"),
    )
    assert policy.recurrent_size == 384
    assert metadata["method"] == "published-v1-factor-overlap-transplant-v2"
    assert len(metadata["source_sha256"]) == 64
    assert metadata["target_action_count"] == RUNNER_ACTION_COUNT_V2

    args.init_checkpoint = checkpoint
    with pytest.raises(ValueError, match="mutually exclusive"):
        initialize_fresh_policy(
            args,
            config,
            torch.device("cpu"),
        )
    args.init_checkpoint = None
    args.resume = True
    with pytest.raises(ValueError, match="cannot be used for a resume"):
        initialize_fresh_policy(
            args,
            config,
            torch.device("cpu"),
        )


def test_seed_namespaces_validation_gates_and_selection_are_fail_closed() -> None:
    require_training_schedule(start=0, env_count=8)
    require_training_schedule(start=TRAINING_SEED_END, env_count=1)
    with pytest.raises(ValueError, match="leave"):
        require_training_schedule(start=TRAINING_SEED_END, env_count=2)
    require_validation_window(7_900, 100)
    with pytest.raises(ValueError, match="namespace"):
        require_validation_window(7_901, 100)

    passing = {tier: ACCEPTANCE_THRESHOLDS[tier] for tier in ALL_TIERS}
    assert acceptance_gate(passing, 0) == 1
    assert acceptance_gate(passing, 1) == 2
    assert acceptance_gate({**passing, 6: 0.84}, 1) == 0
    assert acceptance_gate({1: 1.0}, 1) == 0
    tier, passes, promoted = curriculum_gate(
        current_tier=1,
        rates={1: ACCEPTANCE_THRESHOLDS[1]},
        previous_passes=0,
    )
    assert (tier, passes, promoted) == (1, 1, False)
    tier, passes, promoted = curriculum_gate(
        current_tier=1,
        rates={1: ACCEPTANCE_THRESHOLDS[1]},
        previous_passes=1,
    )
    assert (tier, passes, promoted) == (2, 0, True)

    incomplete_report = {
        "tiers": {
            "1": {
                "success_rate": 1.0,
                "mean_damage": 0.0,
                "mean_duration_seconds": 1.0,
            }
        }
    }
    assert validation_selection_key(incomplete_report)[:2] == (-1.0, -1.0)
    adaptive = smoke_config(
        tiers=ALL_TIERS,
        adaptive_curriculum=True,
        initial_curriculum_tier=1,
    )
    assert selection_validation_tiers(adaptive) == ALL_TIERS
    complete_report = {
        "tiers": {
            str(tier): {
                "success_rate": 0.90 + tier / 100.0,
                "mean_damage": tier / 10.0,
                "mean_duration_seconds": 10.0 + tier,
            }
            for tier in ALL_TIERS
        }
    }
    assert validation_selection_key(complete_report) == pytest.approx(
        (0.91, 0.96, -0.35, -13.5)
    )


def test_training_can_oversample_ghost_without_changing_validation_balance() -> None:
    environment = ScheduledRunnerEnv(
        rank=0,
        env_count=1,
        training_seed_start=0,
        tiers=ALL_TIERS,
        directives=(0, 1, 2, 3),
        schedule_salt=19,
        adaptive_curriculum=True,
        initial_curriculum_tier=3,
        ghost_directive_fraction=0.50,
    )
    try:
        counts = {directive: 0 for directive in ContractDirective}
        for seed in range(10_000):
            _, directive = environment._schedule(seed)
            counts[directive] += 1
    finally:
        environment.close()
    assert counts[ContractDirective.GHOST] / 10_000 == pytest.approx(
        0.50,
        abs=0.02,
    )
    for directive in (
        ContractDirective.STANDARD,
        ContractDirective.SPEED,
        ContractDirective.GREED,
    ):
        assert counts[directive] / 10_000 == pytest.approx(
            1.0 / 6.0,
            abs=0.02,
        )


def test_validation_batches_inference_and_consumes_exact_cursor_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[int, int]] = []

    def observation() -> dict[str, np.ndarray]:
        return {
            "ego": np.zeros(27, dtype=np.float32),
            "objective": np.zeros(8, dtype=np.float32),
            "directive": np.zeros(6, dtype=np.float32),
            "field": np.zeros(8, dtype=np.float32),
            "field_targets": np.zeros((16, 13), dtype=np.float32),
            "field_target_mask": np.zeros(16, dtype=np.int8),
            "local_grid": np.zeros((15, 15, 15), dtype=np.float32),
            "targets": np.zeros((5, 10), dtype=np.float32),
            "target_mask": np.zeros(5, dtype=np.int8),
            "entities": np.zeros((12, 16), dtype=np.float32),
            "entity_mask": np.zeros(12, dtype=np.int8),
            "rays": np.zeros((24, 4), dtype=np.float32),
            "action_mask": np.ones(288, dtype=np.int8),
        }

    class FakeEnv:
        def __init__(self, *, seed: int, tier: int, directive: object):
            del directive
            self.seed = seed
            self.tier = tier
            self.sim = SimpleNamespace(damage_taken=0, elapsed_seconds=1.0)

        def reset(self, *, seed: int, options: dict[str, object]):
            seen.append((int(options["tier"]), seed))
            return observation(), {}

        def step(self, action: int):
            assert action == 0
            return observation(), 0.0, True, False, {
                "is_success": self.seed % 2 == 0,
                "damage": 0,
                "duration_seconds": 1.0,
            }

        def close(self) -> None:
            return None

    class FakePolicy:
        recurrent_size = 256
        training = True

        def eval(self):
            self.training = False
            return self

        def train(self, mode: bool = True):
            self.training = mode
            return self

        def __call__(self, values, hidden):
            batch = values["ego"].shape[0]
            logits = torch.full((batch, 288), -10.0)
            logits[:, 0] = 10.0
            return logits, torch.zeros(batch), hidden

    monkeypatch.setattr("ghostline.runner_train_v2.GhostlineEnvV2", FakeEnv)
    report = validate_runner(
        FakePolicy(),  # type: ignore[arg-type]
        episodes_per_tier=3,
        validation_cursor=17,
        device=torch.device("cpu"),
        tiers=(1, 2),
        batch_size=2,
    )
    assert seen == [
        (tier, validation_seed(tier, cursor))
        for tier in (1, 2)
        for cursor in (17, 18, 19)
    ]
    assert report["tiers"]["1"]["episodes"] == 3
    assert report["tiers"]["2"]["episodes"] == 3


def test_directive_suite_gates_each_tier_by_its_weakest_directive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rates = {
        "standard": 0.96,
        "ghost": 0.91,
        "speed": 0.88,
        "greed": 0.93,
    }

    def fake_validate(*_args, directive, tiers, **_kwargs):
        name = directive.name.lower()
        return {
            "tiers": {
                str(tier): {
                    "episodes": 5,
                    "successes": round(rates[name] * 5),
                    "success_rate": rates[name],
                    "mean_damage": float(int(directive)),
                    "mean_duration_seconds": 10.0 + int(directive),
                }
                for tier in tiers
            }
        }

    monkeypatch.setattr(
        "ghostline.runner_train_v2.validate_runner",
        fake_validate,
    )
    report = validate_runner_suite(
        object(),  # type: ignore[arg-type]
        episodes_per_tier=5,
        validation_cursor=0,
        device=torch.device("cpu"),
        tiers=(1, 2),
    )
    assert report["evaluated_directives"] == (
        "standard",
        "ghost",
        "speed",
        "greed",
    )
    for tier in ("1", "2"):
        assert report["tiers"][tier]["success_rate"] == pytest.approx(0.88)
        assert report["tiers"][tier]["worst_directive"] == "speed"
        assert report["tiers"][tier]["episodes"] == 20


def test_scratch_training_is_explicit_and_last_policy_refreshes_on_resume(
    tmp_path: Path,
) -> None:
    output = tmp_path / "runner"
    base = [
        "--output",
        str(output),
        "--envs",
        "1",
        "--rollout",
        "1",
        "--epochs",
        "1",
        "--minibatch-envs",
        "1",
        "--recurrent-size",
        "256",
        "--sync-envs",
        "--validation-interval",
        "0",
    ]
    with pytest.raises(ValueError, match="requires --published-v1-init"):
        train(build_parser().parse_args([*base, "--max-updates", "1"]))

    first = train(
        build_parser().parse_args(
            [*base, "--allow-scratch", "--max-updates", "1"]
        )
    )
    assert first.name == "last-policy.pt"
    assert not (output / "best.pt").exists()
    first_payload = torch.load(first, map_location="cpu", weights_only=False)
    assert first_payload["metadata"]["updates"] == 1

    resumed = train(
        build_parser().parse_args(
            [*base, "--resume", "--max-updates", "2"]
        )
    )
    assert resumed == first
    resumed_payload = torch.load(
        resumed,
        map_location="cpu",
        weights_only=False,
    )
    assert resumed_payload["metadata"]["updates"] == 2


def test_dry_run_validates_initialization_and_freezes_experiment_manifest(
    tmp_path: Path,
) -> None:
    output = tmp_path / "preflight"
    published = (
        Path(__file__).resolve().parents[1]
        / "models"
        / "ghostline-policy.pt"
    )
    result = train(
        build_parser().parse_args(
            [
                "--output",
                str(output),
                "--published-v1-init",
                str(published),
                "--recurrent-size",
                "384",
                "--envs",
                "2",
                "--sync-envs",
                "--max-updates",
                "3",
                "--dry-run",
                "--cpu",
            ]
        )
    )

    assert result == output / "experiment-manifest.json"
    manifest = json.loads(result.read_text(encoding="utf-8"))
    assert manifest["manifest_contract"] == EXPERIMENT_MANIFEST_CONTRACT
    assert manifest["status"] == "preflight-passed"
    assert manifest["public_environment"] == "GhostlineEnv-v2"
    assert manifest["checkpoint_contract"]["action_count"] == 288
    assert manifest["initialization"]["method"] == (
        "published-v1-factor-overlap-transplant-v2"
    )
    assert len(manifest["initialization"]["source_sha256"]) == 64
    assert manifest["budget"]["max_updates"] == 3
    assert manifest["seed_namespaces"]["final_test"][
        "not_consumed_by_training"
    ]
    assert manifest["hardware"]["selected_device"] == "cpu"
    assert not (output / "latest.pt").exists()
    assert not (output / "last-policy.pt").exists()
