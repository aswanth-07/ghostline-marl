from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch

import ghostline.imitation as imitation
from ghostline.cli import build_parser
from ghostline.curriculum import AdaptiveCurriculum
from ghostline.env import GhostlineEnv
from ghostline.imitation import (
    EpisodeSequenceDataset,
    build_parser as build_imitation_parser,
    behavior_clone_losses,
    collect_trajectories,
    factorized_action_nll,
    recovery_supervision_weights,
    run_dagger,
    train_behavior_clone,
)
from ghostline.model import (
    UniversalGhostlinePolicy,
    current_environment_fingerprint,
    load_policy,
    save_policy,
)
from ghostline.policies import ObservationTeacherPolicy
from ghostline.rnd import RandomNetworkDistillation, decaying_rnd_coefficient
from ghostline.seeds import final_test_seed, validation_seed
from ghostline.torchrl_train import (
    completed_episode_successes,
    make_curriculum_vector_env,
    next_acceptance_passes,
    validation_selection_score,
)


def _tensor(observation: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {key: torch.as_tensor(value).unsqueeze(0) for key, value in observation.items()}


def _write_synthetic_imitation_root(
    root: Path,
    *,
    tiers: tuple[int, ...] = (1,),
    episodes_per_tier: int = 10,
    length: int = 96,
    seed_offset: int = 0,
) -> None:
    root.mkdir(parents=True)
    env = GhostlineEnv(seed=91, tier=1)
    observation, _ = env.reset(seed=91)
    env.close()
    observation_rows = {
        key: np.repeat(np.asarray(observation[key])[None, ...], length, axis=0)
        for key in imitation.OBS_KEYS
    }
    for tier in tiers:
        for episode_index in range(episodes_per_tier):
            seed = tier * 100_000 + seed_offset + episode_index
            actions = np.full(length, 3, dtype=np.int64)
            actions[7::17] = 4
            actions[9] = 12  # move=3, dash=1
            actions[19] = 21  # move=3, pulse=1
            behavior_actions = actions.copy()
            behavior_actions[29] = 1  # explicit policy-induced recovery label
            payload = {
                **observation_rows,
                "action": actions,
                "behavior_action": behavior_actions,
                "reward": np.zeros(length, dtype=np.float32),
                "return_": np.linspace(1.0, 0.0, length, dtype=np.float32),
                "objective_bearing": np.zeros((length, 2), dtype=np.float32),
                "danger": np.zeros(length, dtype=np.float32),
                "tier": np.asarray(tier, dtype=np.int8),
                "requested_tier": np.asarray(tier, dtype=np.int8),
                "seed": np.asarray(seed, dtype=np.int64),
            }
            np.savez_compressed(root / f"tier-{tier}-seed-{seed}.npz", **payload)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format": "ghostline-imitation-v2",
                "observation_contract": "GhostlineEnv-v2",
                "complete": True,
                "environment_stable_during_collection": True,
                "environment_fingerprint": current_environment_fingerprint(),
            }
        ),
        encoding="utf-8",
    )


def test_observation_teacher_is_deterministic_and_legal() -> None:
    env = GhostlineEnv(seed=12, tier=6)
    observation, _ = env.reset(seed=12)
    first = ObservationTeacherPolicy()
    second = ObservationTeacherPolicy()
    for _ in range(20):
        action = first.act(observation)
        assert action == second.act(observation)
        assert observation["action_mask"][action] == 1
        observation, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break
    env.close()


@pytest.mark.parametrize("seed", (1_000_001, 1_000_003, 1_040_000, 1_040_003))
def test_observation_teacher_clears_border_extent_deadlock_seed(seed: int) -> None:
    """Regress border-clearance drift and negative-detour idle deadlocks."""
    env = GhostlineEnv(seed=seed, tier=1)
    observation, _ = env.reset(seed=seed)
    teacher = ObservationTeacherPolicy()
    terminated = truncated = False
    while not (terminated or truncated):
        action = teacher.act(observation)
        teacher.observe_executed_action(action)
        observation, _, terminated, truncated, info = env.step(action)

    assert info["is_success"], info
    assert env.sim.data >= env.sim.level.quota
    env.close()


@pytest.mark.parametrize("seed", (1_040_005, 1_040_006, 1_040_011))
def test_observation_teacher_clears_current_tier6_recovery_regressions(seed: int) -> None:
    env = GhostlineEnv(seed=seed, tier=6)
    observation, _ = env.reset(seed=seed)
    teacher = ObservationTeacherPolicy()
    terminated = truncated = False
    while not (terminated or truncated):
        action = teacher.act(observation)
        teacher.observe_executed_action(action)
        observation, _, terminated, truncated, info = env.step(action)

    assert info["is_success"], info
    assert info["damage"] == 0, info
    env.close()


def _teacher_guard_percept(
    *, confidence: float, grade: float = -1.0, alert: float = 0.8
) -> dict[str, np.ndarray]:
    """Build one public guard row without constructing privileged sim state."""

    entities = np.zeros((12, 13), dtype=np.float32)
    entities[0, :3] = (1.0, -1.0, -1.0)
    entities[0, 3:5] = (60.0 / 390.0, 0.0)
    entities[0, 5] = 60.0 / 390.0 * 2.0 - 1.0
    entities[0, 8:10] = (0.0, -1.0)  # facing toward the player
    entities[0, 10] = alert * 2.0 - 1.0
    entities[0, 11] = confidence * 2.0 - 1.0
    entities[0, 12] = grade
    mask = np.zeros(12, dtype=np.int8)
    mask[0] = 1
    return {"entities": entities, "entity_mask": mask}


def test_teacher_last_seen_pressure_expires_at_public_confidence_floor() -> None:
    fresh = ObservationTeacherPolicy._threats(_teacher_guard_percept(confidence=0.90))[1]
    older = ObservationTeacherPolicy._threats(_teacher_guard_percept(confidence=0.70))[1]
    expired = ObservationTeacherPolicy._threats(_teacher_guard_percept(confidence=0.51))[1]

    assert fresh > older > expired
    assert expired == pytest.approx(0.0)


def test_teacher_uses_coarse_audio_without_inventing_a_guard_view_cone() -> None:
    observation = _teacher_guard_percept(confidence=0.30)
    observation["entities"][0, 8:10] = 0.0  # audible rows expose unknown facing
    threats, immediate, _electronics, pursuit = ObservationTeacherPolicy._threats(observation)

    assert threats[0][2] == pytest.approx(0.0)
    assert immediate > 0.0
    assert pursuit > 0.0


def test_teacher_decodes_return_mode_and_explicit_guard_grade_semantically() -> None:
    returning = ObservationTeacherPolicy._threats(
        _teacher_guard_percept(confidence=1.0, grade=-1.0, alert=1.0)
    )[0][0][1]
    standard = ObservationTeacherPolicy._threats(
        _teacher_guard_percept(confidence=1.0, grade=-1.0, alert=0.8)
    )[0][0][1]
    elite = ObservationTeacherPolicy._threats(
        _teacher_guard_percept(confidence=1.0, grade=1.0, alert=0.8)
    )[0][0][1]

    assert returning < standard < elite


def test_dagger_teacher_memory_tracks_executed_behavior_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_env = GhostlineEnv(seed=3, tier=1)
    observation, _ = real_env.reset(seed=3)
    real_env.close()
    observed_actions: list[int] = []

    class SpyTeacher:
        def act(self, _observation):
            return 1

        def observe_executed_action(self, action: int) -> None:
            observed_actions.append(action)

    class OneStepEnv:
        tier = 1

        def __init__(self, **_kwargs):
            pass

        def reset(self, **_kwargs):
            return observation, {"tier": 1}

        def step(self, action: int):
            assert action == 3
            return observation, 0.0, True, False, {
                "is_success": False,
                "fail_reason": "contract_expired",
            }

        def close(self) -> None:
            pass

    class Behavior:
        def act(self, _observation, hidden, **_kwargs):
            return 3, hidden

    monkeypatch.setattr(imitation, "ObservationTeacherPolicy", SpyTeacher)
    monkeypatch.setattr(imitation, "GhostlineEnv", OneStepEnv)
    result = imitation._collect_episode(
        tmp_path,
        requested_tier=1,
        seed=3,
        lesson=0,
        behavior=Behavior(),
        teacher_probability=0.0,
        rng=np.random.default_rng(0),
    )

    assert result.transitions == 1
    assert observed_actions == [3]


def test_recurrent_training_vector_env_uses_same_step_autoreset() -> None:
    envs = make_curriculum_vector_env(
        env_count=1,
        curriculum=AdaptiveCurriculum(),
        training_lesson=1,
        fixed_tier=1,
        async_envs=False,
    )
    try:
        assert envs.autoreset_mode is gym.vector.AutoresetMode.SAME_STEP
    finally:
        envs.close()


def test_vector_terminal_info_supports_current_nested_and_legacy_layouts() -> None:
    done = np.asarray((True, False, True), dtype=bool)
    current = {
        "final_info": {
            "is_success": np.asarray((True, False, False)),
            "_is_success": np.asarray((True, False, True)),
        },
        "_final_info": np.asarray((True, False, True)),
    }
    assert completed_episode_successes(current, done) == [1.0, 0.0]

    legacy = {
        "final_info": np.asarray(
            ({"is_success": True}, None, {"is_success": False}),
            dtype=object,
        ),
        "_final_info": np.asarray((True, False, True)),
    }
    assert completed_episode_successes(legacy, done) == [1.0, 0.0]


@pytest.mark.parametrize("hidden_size", (256, 384))
def test_policy_recurrent_width_is_checkpointed(tmp_path: Path, hidden_size: int) -> None:
    env = GhostlineEnv(seed=2, tier=1)
    observation, _ = env.reset(seed=2)
    policy = UniversalGhostlinePolicy(recurrent_size=hidden_size)
    _, _, hidden = policy(_tensor(observation))
    assert hidden is not None and hidden.shape == (1, 1, hidden_size)
    checkpoint = tmp_path / f"policy-{hidden_size}.pt"
    save_policy(policy, checkpoint, purpose="test")
    restored = load_policy(checkpoint)
    assert restored.recurrent_size == hidden_size
    env.close()


def test_policy_checkpoint_rejects_missing_or_stale_environment_fingerprint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "policy.pt"
    save_policy(UniversalGhostlinePolicy(recurrent_size=256), checkpoint, purpose="contract-test")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert payload["environment_fingerprint"] == current_environment_fingerprint()

    payload["environment_fingerprint"] = "stale"
    payload["metadata"]["environment_fingerprint"] = "stale"
    torch.save(payload, checkpoint)
    with pytest.raises(RuntimeError, match="stale environment fingerprint"):
        load_policy(checkpoint)

    payload.pop("environment_fingerprint")
    payload["metadata"].pop("environment_fingerprint")
    torch.save(payload, checkpoint)
    with pytest.raises(RuntimeError, match="no environment fingerprint"):
        load_policy(checkpoint)


def test_behavior_clone_can_initialize_from_prior_policy(tmp_path: Path) -> None:
    data = tmp_path / "teacher"
    collect_trajectories(data, tiers=(1,), episodes_per_tier=2, lesson=1, overwrite=True)
    initial = UniversalGhostlinePolicy(recurrent_size=256)
    checkpoint = tmp_path / "initial.pt"
    save_policy(initial, checkpoint, purpose="dagger-init-test")
    output = tmp_path / "continued"
    trained = train_behavior_clone(
        (data,),
        output,
        updates=1,
        batch_size=1,
        sequence_length=2,
        burn_in=0,
        recurrent_size=256,
        init_checkpoint=checkpoint,
        resume=False,
    )
    assert trained.exists()
    assert trained.name == "latest.pt"
    assert (output / "best.pt").exists()
    best_payload = torch.load(output / "best.pt", map_location="cpu", weights_only=False)
    assert best_payload["metadata"]["stage"] == (
        "behavior_cloning_held_out_episode_validation"
    )
    payload = torch.load(trained, map_location="cpu", weights_only=False)
    assert np.isfinite(payload["validation_loss"])
    assert payload["data_split_digest"]
    assert payload["numpy_rng_state"]
    split_report = json.loads((output / "data-split.json").read_text(encoding="utf-8"))
    assert set(split_report["train_file_ids"]).isdisjoint(split_report["validation_file_ids"])
    assert split_report["split"]["train_episodes"] == 1
    assert split_report["split"]["validation_episodes"] == 1
    assert split_report["validation_windows"] == 128
    assert split_report["validation_input_tensor_bytes"] > 0
    assert split_report["validation_input_tensor_mib"] > 0
    validation_sample = split_report["held_out_validation_sample"]
    assert validation_sample["requested_counts"] == {
        "uniform": 64,
        "endpoint": 13,
        "action_change": 19,
        "dash": 6,
        "pulse": 13,
        "recovery": 13,
    }
    assert set(validation_sample["episode_ids"]) <= set(
        split_report["validation_file_ids"]
    )
    training_state = json.loads((output / "training-state.json").read_text(encoding="utf-8"))
    assert training_state["update"] == 1
    assert training_state["data_split_digest"] == payload["data_split_digest"]
    assert training_state["split_sizes"] == {
        "train_episodes": 1,
        "held_out_episodes": 1,
    }
    for section in ("train", "held_out"):
        assert {
            "loss",
            "action_accuracy",
            "move_accuracy",
            "dash_accuracy",
            "pulse_accuracy",
            "pulse_positive_precision",
            "pulse_positive_recall",
            "recovery_action_accuracy",
        } <= training_state[section].keys()


def test_episode_split_is_deterministic_disjoint_and_root_tier_representative(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base"
    recovery = tmp_path / "recovery"
    _write_synthetic_imitation_root(base, tiers=(1, 2), episodes_per_tier=10)
    _write_synthetic_imitation_root(
        recovery, tiers=(1, 2), episodes_per_tier=10, seed_offset=10_000
    )
    kwargs = {
        "validation_fraction": 0.2,
        "split_seed": 37,
        "latest_root_fraction": 0.5,
    }
    train = EpisodeSequenceDataset((base, recovery), split="train", **kwargs)
    held_out = EpisodeSequenceDataset((base, recovery), split="validation", **kwargs)
    repeated = EpisodeSequenceDataset((base, recovery), split="validation", **kwargs)
    all_episodes = EpisodeSequenceDataset((base, recovery), split="all", **kwargs)

    assert set(train.file_ids).isdisjoint(held_out.file_ids)
    assert set(train.file_ids) | set(held_out.file_ids) == set(all_episodes.file_ids)
    assert held_out.file_ids == repeated.file_ids
    assert train.split_report["split_digest"] == held_out.split_report["split_digest"]
    assert held_out.split_report["split_digest"] == repeated.split_report["split_digest"]
    for group in train.split_report["groups"]:
        assert group["representative"] is True
        assert group["train_episodes"] == 8
        assert group["validation_episodes"] == 2
    different_seed = EpisodeSequenceDataset(
        (base, recovery),
        split="validation",
        validation_fraction=0.2,
        split_seed=38,
        latest_root_fraction=0.5,
    )
    assert held_out.file_ids != different_seed.file_ids


def test_stratified_windows_are_deterministic_and_cover_tail_decisions(
    tmp_path: Path,
) -> None:
    root = tmp_path / "stratified"
    _write_synthetic_imitation_root(root, episodes_per_tier=8)
    first = EpisodeSequenceDataset((root,))
    second = EpisodeSequenceDataset((root,))
    sample_kwargs = {
        "batch_size": 100,
        "sequence_length": 16,
        "burn_in": 8,
        "device": torch.device("cpu"),
        "uniform_fraction": 0.5,
    }
    first_observations, first_labels = first.sample(
        **sample_kwargs, rng=np.random.default_rng(713)
    )
    second_observations, second_labels = second.sample(
        **sample_kwargs, rng=np.random.default_rng(713)
    )
    for key in first_observations:
        assert torch.equal(first_observations[key], second_observations[key])
    for key in first_labels:
        assert torch.equal(first_labels[key], second_labels[key])
    assert first.last_sample_metrics == second.last_sample_metrics
    metrics = first.last_sample_metrics
    assert metrics["requested_counts"] == {
        "uniform": 50,
        "endpoint": 10,
        "action_change": 15,
        "dash": 5,
        "pulse": 10,
        "recovery": 10,
    }
    assert metrics["actual_counts"] == metrics["requested_counts"]
    assert metrics["fallbacks"] == {}
    assert first_labels["valid"].shape == (24, 100)
    assert torch.all(first_labels["valid"].sum(dim=0) == 16)

    paths = {f"0:{path.name}": path for path in first.files}
    for batch_index, (episode_id, stratum, start, anchor) in enumerate(
        zip(
            metrics["episode_ids"],
            metrics["actual_strata"],
            metrics["training_starts"],
            metrics["anchors"],
        )
    ):
        if stratum == "uniform":
            assert anchor is None
            assert torch.all(first_labels["anchor_weight"][:, batch_index] == 1)
            continue
        assert start <= anchor < start + 16
        anchor_position = anchor - max(0, start - 8)
        assert first_labels["anchor_weight"][anchor_position, batch_index] == pytest.approx(
            imitation.ANCHOR_LOSS_WEIGHTS[stratum]
        )
        assert first_labels["valid"][anchor_position, batch_index] == 1
        with np.load(paths[episode_id], allow_pickle=False) as episode:
            action = episode["action"]
            behavior_action = episode["behavior_action"]
            if stratum == "endpoint":
                assert anchor == len(action) - 1
            elif stratum == "action_change":
                assert anchor > 0 and action[anchor] != action[anchor - 1]
            elif stratum == "dash":
                assert (action[anchor] // 9) % 2 == 1
            elif stratum == "pulse":
                assert action[anchor] // 18 == 1
            else:
                assert stratum == "recovery"
                assert action[anchor] != behavior_action[anchor]

    batch_sixteen = EpisodeSequenceDataset._window_schedule(
        16,
        stratified=True,
        uniform_fraction=0.5,
        rng=np.random.default_rng(19),
    )
    assert {stratum: batch_sixteen.count(stratum) for stratum in imitation.WINDOW_STRATA} == {
        "uniform": 8,
        "endpoint": 2,
        "action_change": 2,
        "dash": 1,
        "pulse": 2,
        "recovery": 1,
    }


def test_strict_episode_split_rejects_duplicates_and_singleton_groups(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first"
    duplicate = tmp_path / "duplicate"
    _write_synthetic_imitation_root(first, episodes_per_tier=2)
    _write_synthetic_imitation_root(duplicate, episodes_per_tier=2)
    with pytest.raises(RuntimeError, match="Duplicate imitation episode identity"):
        EpisodeSequenceDataset((first, duplicate), split="train")

    singleton = tmp_path / "singleton"
    _write_synthetic_imitation_root(singleton, episodes_per_tier=1)
    with pytest.raises(RuntimeError, match="at least two episodes"):
        EpisodeSequenceDataset(
            (singleton,),
            split="train",
            require_representative_split=True,
        )


def test_factorized_action_loss_matches_public_action_ordering() -> None:
    actions = torch.arange(36)
    logits = torch.full((36, 36), -12.0)
    logits[actions, actions] = 12.0
    move_nll, dash_nll, pulse_nll = factorized_action_nll(logits, actions)
    assert torch.max(move_nll) < 1e-6
    assert torch.max(dash_nll) < 1e-6
    assert torch.max(pulse_nll) < 1e-6

    target = torch.tensor([7 + 9 + 18])  # move=7, dash=1, pulse=1
    component_logits = torch.full((1, 36), -10.0)
    component_logits[0, int(target)] = 10.0
    move, dash, pulse = factorized_action_nll(component_logits, target)
    assert move.item() < 1e-6
    assert dash.item() < 1e-6
    assert pulse.item() < 1e-6


def test_factorized_action_loss_respects_legal_mask_and_has_finite_gradients() -> None:
    raw_logits = torch.randn(2, 3, 36, requires_grad=True)
    legal = torch.zeros(2, 3, 36, dtype=torch.bool)
    actions = torch.tensor([[3, 12, 21], [8, 17, 35]])
    legal.scatter_(-1, actions.unsqueeze(-1), True)
    legal[..., :9] = True
    masked_logits = raw_logits.masked_fill(~legal, -1e9)
    component_losses = factorized_action_nll(masked_logits, actions)
    loss = sum(component.mean() for component in component_losses)
    loss.backward()
    assert torch.isfinite(loss)
    assert raw_logits.grad is not None
    assert torch.isfinite(raw_logits.grad).all()


def test_recovery_move_corrections_receive_fourfold_joint_and_move_weight() -> None:
    teacher_actions = torch.tensor([[3], [3]])
    behavior_actions = torch.tensor([[3], [1]])
    weights, recovery = recovery_supervision_weights(teacher_actions, behavior_actions)
    assert weights[:, 0].tolist() == [1.0, 4.0]
    assert recovery[:, 0].tolist() == [0.0, 1.0]

    logits = torch.full((2, 1, 36), -4.0)
    logits[0, 0, 3] = 4.0
    logits[1, 0, 1] = 4.0
    move_nll, _, _ = factorized_action_nll(logits, teacher_actions)
    labels = {
        "action": teacher_actions,
        "behavior_action": behavior_actions,
        "valid": torch.ones(2, 1),
        "objective_bearing": torch.zeros(2, 1, 2),
        "danger": torch.zeros(2, 1),
        "return_": torch.zeros(2, 1),
    }
    metrics = behavior_clone_losses(
        logits,
        torch.zeros(2, 1),
        torch.zeros(2, 1, 2),
        torch.full((2, 1), 0.5),
        labels,
    )
    expected_weighted_move = (move_nll[0, 0] + 4.0 * move_nll[1, 0]) / 5.0
    assert metrics["move_nll"] == pytest.approx(float(expected_weighted_move))
    assert metrics["recovery_fraction"] == pytest.approx(0.5)


def test_component_only_recovery_and_positive_pulse_metrics_are_visible() -> None:
    teacher_actions = torch.tensor([[21]])  # move=3, pulse=1
    behavior_actions = torch.tensor([[3]])  # same move, pulse omitted
    logits = torch.full((1, 1, 36), -8.0)
    logits[0, 0, 3] = 8.0
    labels = {
        "action": teacher_actions,
        "behavior_action": behavior_actions,
        "valid": torch.ones(1, 1),
        "objective_bearing": torch.zeros(1, 1, 2),
        "danger": torch.zeros(1, 1),
        "return_": torch.zeros(1, 1),
    }
    metrics = behavior_clone_losses(
        logits,
        torch.zeros(1, 1),
        torch.zeros(1, 1, 2),
        torch.full((1, 1), 0.5),
        labels,
    )
    assert metrics["recovery_fraction"] == pytest.approx(1.0)
    assert metrics["move_recovery_fraction"] == pytest.approx(0.0)
    assert metrics["recovery_action_accuracy"] == pytest.approx(0.0)
    assert metrics["recovery_move_accuracy"] == pytest.approx(1.0)
    assert metrics["pulse_positive_count"] == pytest.approx(1.0)
    assert metrics["pulse_positive_recall"] == pytest.approx(0.0)


def test_pulse_anchor_has_bounded_material_loss_weight() -> None:
    actions = torch.tensor([[21], [3]])
    logits = torch.full((2, 1, 36), -8.0)
    logits[:, 0, 3] = 8.0
    base_labels = {
        "action": actions,
        "behavior_action": actions,
        "valid": torch.ones(2, 1),
        "objective_bearing": torch.zeros(2, 1, 2),
        "danger": torch.zeros(2, 1),
        "return_": torch.zeros(2, 1),
    }
    unweighted = behavior_clone_losses(
        logits,
        torch.zeros(2, 1),
        torch.zeros(2, 1, 2),
        torch.full((2, 1), 0.5),
        base_labels,
    )
    weighted = behavior_clone_losses(
        logits,
        torch.zeros(2, 1),
        torch.zeros(2, 1, 2),
        torch.full((2, 1), 0.5),
        {**base_labels, "anchor_weight": torch.tensor([[6.0], [1.0]])},
    )
    assert weighted["pulse_nll"] > unweighted["pulse_nll"] * 1.6
    assert weighted["imitation_loss"] > unweighted["imitation_loss"] * 1.6
    assert weighted["mean_priority_weight"] == pytest.approx(3.5)
    assert weighted["pulse_positive_recall"] == pytest.approx(0.0)
    assert weighted["pulse_positive_precision"] == pytest.approx(0.0)


def test_latest_recovery_root_supplies_half_of_sampled_sequences(tmp_path: Path) -> None:
    base = tmp_path / "base"
    recovery = tmp_path / "recovery"
    base.mkdir()
    recovery.mkdir()
    for index in range(20):
        (base / f"tier-1-seed-{index}.npz").write_bytes(b"base")
    for index in range(2):
        (recovery / f"tier-1-seed-{100 + index}.npz").write_bytes(b"recovery")
    for root in (base, recovery):
        (root / "manifest.json").write_text(
            json.dumps(
                {
                    "format": "ghostline-imitation-v2",
                    "observation_contract": "GhostlineEnv-v2",
                    "complete": True,
                    "environment_stable_during_collection": True,
                    "environment_fingerprint": current_environment_fingerprint(),
                }
            ),
            encoding="utf-8",
        )
    dataset = EpisodeSequenceDataset((base, recovery), latest_root_fraction=0.5)
    rng = np.random.default_rng(17)
    recovery_samples = sum(dataset._sample_path(rng).parent == recovery for _ in range(2_000))
    assert recovery_samples / 2_000 == pytest.approx(0.5, abs=0.035)


def test_imitation_dataset_rejects_unversioned_and_stale_roots(tmp_path: Path) -> None:
    root = tmp_path / "teacher"
    root.mkdir()
    (root / "tier-1-seed-1.npz").write_bytes(b"episode")
    with pytest.raises(FileNotFoundError, match="no manifest"):
        EpisodeSequenceDataset((root,))

    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format": "ghostline-imitation-v2",
                "observation_contract": "GhostlineEnv-v2",
                "complete": True,
                "environment_stable_during_collection": True,
                "environment_fingerprint": "stale",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="Stale imitation dataset"):
        EpisodeSequenceDataset((root,))


def test_teacher_stall_detection_uses_player_motion_not_clipped_goal_distance() -> None:
    env = GhostlineEnv(seed=31, tier=6)
    observation, _ = env.reset(seed=31)
    observation = {key: np.asarray(value).copy() for key, value in observation.items()}
    observation["objective"][3] = 1.0
    observation["objective"][4:6] = (1.0, 0.0)
    moving = ObservationTeacherPolicy()
    moving.last_move = 3
    moving.last_position = observation["ego"][19:21].copy()
    for _ in range(10):
        observation["ego"][19] += 0.01
        observation["ego"][0] = 0.5
        moving.act(observation)
    assert moving.stalled_steps == 0

    stalled = ObservationTeacherPolicy()
    stalled.last_move = 3
    stalled.last_position = observation["ego"][19:21].copy()
    observation["ego"][:2] = 0.0
    for _ in range(4):
        stalled.act(observation)
    assert stalled.stalled_steps == 4
    env.close()


def test_dagger_resume_reuses_prior_datasets_without_rerunning_rounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "teacher"
    prior_data = tmp_path / "dagger" / "round-1" / "data"
    base.mkdir()
    prior_data.mkdir(parents=True)
    (base / "tier-1-seed-1.npz").write_bytes(b"base")
    prior_marker = prior_data / "tier-1-seed-2.npz"
    prior_marker.write_bytes(b"prior")
    initial_checkpoint = tmp_path / "dagger" / "round-1" / "model" / "best.pt"
    initial_checkpoint.parent.mkdir(parents=True)
    initial_checkpoint.write_bytes(b"checkpoint")
    calls: dict[str, object] = {}

    def fake_collect(output: Path, **kwargs: object) -> None:
        calls["collect_output"] = output
        calls["collect_kwargs"] = kwargs
        output.mkdir(parents=True)
        (output / "tier-1-seed-3.npz").write_bytes(b"round-2")

    def fake_train(datasets: object, output: Path, **kwargs: object) -> Path:
        calls["datasets"] = tuple(datasets)  # type: ignore[arg-type]
        calls["train_output"] = output
        calls["train_kwargs"] = kwargs
        output.mkdir(parents=True)
        checkpoint = output / "best.pt"
        checkpoint.write_bytes(b"trained")
        return checkpoint

    monkeypatch.setattr(imitation, "collect_trajectories", fake_collect)
    monkeypatch.setattr(imitation, "train_behavior_clone", fake_train)
    result = run_dagger(
        base,
        tmp_path / "dagger",
        initial_checkpoint,
        rounds=2,
        start_round=2,
        episodes_per_tier=20,
        updates_per_round=3_000,
        beta_start=0.5,
        beta_decay=0.5,
        recurrent_size=256,
        collection_device="cpu",
        collection_workers=12,
        device="cuda",
    )

    assert result == tmp_path / "dagger" / "round-2" / "model" / "best.pt"
    assert prior_marker.read_bytes() == b"prior"
    assert calls["collect_output"] == tmp_path / "dagger" / "round-2" / "data"
    collect_kwargs = calls["collect_kwargs"]
    assert isinstance(collect_kwargs, dict)
    assert collect_kwargs["seed_start"] == 20_000
    assert collect_kwargs["teacher_probability"] == pytest.approx(0.25)
    assert collect_kwargs["behavior_checkpoint"] == initial_checkpoint
    assert collect_kwargs["collection_device"] == "cpu"
    assert collect_kwargs["workers"] == 12
    assert calls["datasets"] == (
        base,
        tmp_path / "dagger" / "round-1" / "data",
        tmp_path / "dagger" / "round-2" / "data",
    )
    train_kwargs = calls["train_kwargs"]
    assert isinstance(train_kwargs, dict)
    assert train_kwargs["updates"] == 3_000
    assert train_kwargs["init_checkpoint"] == initial_checkpoint
    assert train_kwargs["device"] == "cuda"
    assert train_kwargs["sequence_length"] == 64
    assert train_kwargs["burn_in"] == 32
    assert train_kwargs["validation_windows"] == 128


def test_dagger_resume_requires_every_prior_round_dataset(tmp_path: Path) -> None:
    base = tmp_path / "teacher"
    base.mkdir()
    (base / "tier-1-seed-1.npz").write_bytes(b"base")
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    with pytest.raises(FileNotFoundError, match="prior round 1 dataset"):
        run_dagger(
            base,
            tmp_path / "dagger",
            checkpoint,
            rounds=2,
            start_round=2,
            episodes_per_tier=1,
            updates_per_round=1,
        )


def test_dagger_round_seed_subranges_are_disjoint_and_training_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "teacher"
    base.mkdir()
    (base / "tier-1-seed-1.npz").write_bytes(b"base")
    checkpoint = tmp_path / "bc.pt"
    checkpoint.write_bytes(b"checkpoint")
    seed_starts: list[int] = []
    teacher_probabilities: list[float] = []

    def fake_collect(output: Path, **kwargs: object) -> None:
        seed_starts.append(int(kwargs["seed_start"]))
        teacher_probabilities.append(float(kwargs["teacher_probability"]))
        output.mkdir(parents=True)
        (output / "tier-1-seed-1.npz").write_bytes(b"round")

    def fake_train(datasets: object, output: Path, **kwargs: object) -> Path:
        del datasets, kwargs
        output.mkdir(parents=True)
        result = output / "best.pt"
        result.write_bytes(b"checkpoint")
        return result

    monkeypatch.setattr(imitation, "collect_trajectories", fake_collect)
    monkeypatch.setattr(imitation, "train_behavior_clone", fake_train)
    run_dagger(
        base,
        tmp_path / "dagger",
        checkpoint,
        rounds=3,
        episodes_per_tier=20,
        updates_per_round=1,
        beta_start=0.0,
    )
    assert seed_starts == [10_000, 20_000, 30_000]
    assert teacher_probabilities == [0.0, 0.0, 0.0]
    tier_six_ranges = [
        range(start + 600_000, start + 600_000 + 20)
        for start in seed_starts
    ]
    assert all(set(left).isdisjoint(right) for left, right in zip(tier_six_ranges, tier_six_ranges[1:]))
    assert max(seed for seeds in tier_six_ranges for seed in seeds) < 700_000


def test_dagger_rejects_colliding_or_non_training_seed_schedules(tmp_path: Path) -> None:
    base = tmp_path / "teacher"
    base.mkdir()
    (base / "tier-1-seed-1.npz").write_bytes(b"base")
    checkpoint = tmp_path / "bc.pt"
    checkpoint.write_bytes(b"checkpoint")
    with pytest.raises(ValueError, match="round seed subranges disjoint"):
        run_dagger(
            base,
            tmp_path / "dagger",
            checkpoint,
            rounds=1,
            episodes_per_tier=10_001,
            updates_per_round=1,
        )
    with pytest.raises(ValueError, match="outside the training namespace"):
        run_dagger(
            base,
            tmp_path / "dagger",
            checkpoint,
            rounds=40,
            start_round=40,
            episodes_per_tier=1,
            updates_per_round=1,
        )


def test_recurrent_sequence_resets_memory_at_episode_boundaries() -> None:
    env = GhostlineEnv(seed=4, tier=1)
    observation, _ = env.reset(seed=4)
    observations = []
    for _ in range(3):
        observations.append(_tensor(observation))
        observation, _, _, _, _ = env.step(3)
    sequence = {
        key: torch.stack([item[key][0] for item in observations]).unsqueeze(1)
        for key in observations[0]
    }
    policy = UniversalGhostlinePolicy(recurrent_size=256).eval()
    resets = torch.tensor([[0.0], [1.0], [0.0]])
    sequence_logits, _, _ = policy.forward_sequence(sequence, reset_mask=resets)
    hidden = None
    for tick, item in enumerate(observations):
        if resets[tick, 0]:
            hidden = None
        logits, _, hidden = policy(item, hidden)
        assert torch.allclose(sequence_logits[tick], logits, atol=1e-5)
    env.close()


def test_teacher_collection_manifest_and_sequence_sampler(tmp_path: Path) -> None:
    data = tmp_path / "teacher"
    summary = collect_trajectories(
        data,
        tiers=(1,),
        episodes_per_tier=1,
        lesson=1,
        overwrite=True,
    )
    assert summary.episodes == 1
    assert summary.transitions > 0
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["observation_contract"] == "GhostlineEnv-v2"
    assert manifest["teacher_uses_privileged_state"] is False
    assert manifest["collection_device"] == "cpu"
    assert manifest["behavior_policy_used"] is False
    dataset = EpisodeSequenceDataset((data,))
    observations, labels = dataset.sample(
        batch_size=2,
        sequence_length=4,
        rng=np.random.default_rng(0),
        device=torch.device("cpu"),
    )
    assert observations["objective"].shape == (4, 2, 8)
    assert labels["action"].shape == (4, 2)
    assert torch.all(labels["valid"] == 1)
    burn_observations, burn_labels = dataset.sample(
        batch_size=1,
        sequence_length=4,
        burn_in=2,
        rng=np.random.default_rng(3),
        device=torch.device("cpu"),
    )
    assert burn_observations["objective"].shape == (6, 1, 8)
    assert burn_labels["valid"].sum() <= 4


def test_parallel_success_only_teacher_collection_uses_actual_lesson_tier(tmp_path: Path) -> None:
    data = tmp_path / "parallel-teacher"
    summary = collect_trajectories(
        data,
        tiers=(6,),
        episodes_per_tier=2,
        lesson=1,
        workers=2,
        success_only=True,
        overwrite=True,
    )
    files = sorted(data.glob("tier-*-seed-*.npz"))
    assert summary.episodes == 2
    assert summary.behavior_successes == 2
    assert len(files) == 2
    assert all(path.name.startswith("tier-1-seed-") for path in files)
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["workers"] == 2
    assert manifest["success_only"] is True
    assert manifest["attempted_episodes"] >= 2
    assert manifest["discarded_episodes"] == manifest["attempted_episodes"] - 2
    assert manifest["tier_counts"][0]["requested_tier"] == 6
    assert manifest["tier_counts"][0]["actual_tiers"] == [1]
    assert {record["tier"] for record in manifest["records"]} == {1}
    assert {record["requested_tier"] for record in manifest["records"]} == {6}
    with np.load(files[0], allow_pickle=False) as episode:
        assert int(episode["tier"]) == 1
        assert int(episode["requested_tier"]) == 6


def test_success_only_discards_failed_attempts_and_counts_them(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "filtered-teacher"

    def fake_collect(job: imitation._PureTeacherJob) -> imitation._EpisodeResult:
        success = int(job.seed % 100_000 == 1)
        path = None
        if success:
            path = job.output / f"tier-1-seed-{job.seed}.npz"
            np.savez_compressed(path, action=np.asarray([0], dtype=np.int64))
        return imitation._EpisodeResult(
            requested_tier=job.requested_tier,
            tier=1,
            seed=job.seed,
            transitions=3,
            behavior_success=success,
            fail_reason="none" if success else "timeout",
            path=path,
        )

    monkeypatch.setattr(imitation, "_collect_pure_teacher_episode", fake_collect)
    summary = collect_trajectories(
        data,
        tiers=(1,),
        episodes_per_tier=1,
        success_only=True,
        max_attempts_per_tier=3,
    )
    assert summary.episodes == 1
    assert summary.attempted_episodes == 2
    assert summary.discarded_episodes == 1
    assert len(list(data.glob("*.npz"))) == 1
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert manifest["attempted_episodes"] == 2
    assert manifest["discarded_episodes"] == 1
    assert manifest["discarded_transitions"] == 3


def test_success_only_attempt_limit_writes_incomplete_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data = tmp_path / "bounded-teacher"

    def always_fail(job: imitation._PureTeacherJob) -> imitation._EpisodeResult:
        return imitation._EpisodeResult(
            requested_tier=job.requested_tier,
            tier=job.requested_tier,
            seed=job.seed,
            transitions=2,
            behavior_success=0,
            fail_reason="timeout",
            path=None,
        )

    monkeypatch.setattr(imitation, "_collect_pure_teacher_episode", always_fail)
    with pytest.raises(RuntimeError, match="exhausted its deterministic attempt limit"):
        collect_trajectories(
            data,
            tiers=(1,),
            episodes_per_tier=1,
            success_only=True,
            max_attempts_per_tier=2,
        )
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is False
    assert manifest["attempted_episodes"] == 2
    assert manifest["discarded_episodes"] == 2
    assert manifest["shortfalls"] == [
        {"requested_tier": 1, "requested_episodes": 1, "retained_episodes": 0}
    ]


def test_collection_fails_closed_when_environment_changes_mid_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fingerprints = iter(("before", "after"))
    monkeypatch.setattr(
        imitation,
        "training_environment_fingerprint",
        lambda: next(fingerprints),
    )

    def fake_collect(job: imitation._PureTeacherJob) -> imitation._EpisodeResult:
        path = job.output / f"tier-1-seed-{job.seed}.npz"
        np.savez_compressed(path, action=np.asarray([0], dtype=np.int64))
        return imitation._EpisodeResult(
            requested_tier=1,
            tier=1,
            seed=job.seed,
            transitions=1,
            behavior_success=1,
            fail_reason="none",
            path=path,
        )

    monkeypatch.setattr(imitation, "_collect_pure_teacher_episode", fake_collect)
    output = tmp_path / "moving-environment"
    with pytest.raises(RuntimeError, match="changed during trajectory collection"):
        collect_trajectories(output, tiers=(1,), episodes_per_tier=1)

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is False
    assert manifest["environment_fingerprint"] == "before"
    assert manifest["environment_stable_during_collection"] is False


def test_mixed_collection_requires_checkpoint_before_creating_output(tmp_path: Path) -> None:
    output = tmp_path / "missing-policy"
    with pytest.raises(ValueError, match="requires behavior_checkpoint"):
        collect_trajectories(
            output,
            tiers=(1,),
            episodes_per_tier=1,
            teacher_probability=0.5,
        )
    assert not output.exists()


def test_parallel_behavior_collection_requires_cpu_before_creating_output(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "policy.pt"
    checkpoint.write_bytes(b"not loaded because validation fails first")
    output = tmp_path / "gpu-workers"
    with pytest.raises(ValueError, match="requires a CPU collection_device"):
        collect_trajectories(
            output,
            tiers=(1,),
            episodes_per_tier=1,
            behavior_checkpoint=checkpoint,
            teacher_probability=0.5,
            collection_device="cuda",
            workers=2,
        )
    assert not output.exists()


def test_dagger_mixture_rng_is_episode_local_and_repeatable() -> None:
    first = imitation._episode_mixture_rng(requested_tier=3, seed=310_007, lesson=0)
    repeated = imitation._episode_mixture_rng(requested_tier=3, seed=310_007, lesson=0)
    other_episode = imitation._episode_mixture_rng(requested_tier=3, seed=310_008, lesson=0)
    first_values = first.random(32)
    assert np.array_equal(first_values, repeated.random(32))
    assert not np.array_equal(first_values, other_episode.random(32))


@pytest.mark.parametrize(
    ("seed_start", "match"),
    ((-1, "non-negative"), (400_000, "outside the training namespace")),
)
def test_imitation_collection_rejects_non_training_seed_schedules(
    tmp_path: Path, seed_start: int, match: str
) -> None:
    output = tmp_path / "invalid-seeds"
    with pytest.raises(ValueError, match=match):
        collect_trajectories(
            output,
            tiers=(6,),
            episodes_per_tier=1,
            seed_start=seed_start,
        )
    assert not output.exists()


def test_rnd_bonus_is_finite_and_schedule_decays() -> None:
    env = GhostlineEnv(seed=8, tier=3)
    observation, _ = env.reset(seed=8)
    tensors = _tensor(observation)
    rnd = RandomNetworkDistillation()
    bonus = rnd.intrinsic_reward(tensors)
    loss = rnd.predictor_loss(tensors, update_fraction=1.0)
    assert bonus.shape == (1,)
    assert torch.isfinite(bonus).all() and torch.isfinite(loss)
    assert decaying_rnd_coefficient(0, initial=0.1, decay_steps=100) == pytest.approx(0.1)
    assert decaying_rnd_coefficient(100, initial=0.1, decay_steps=100) == pytest.approx(0.005)
    env.close()


def test_rl_cli_exposes_hybrid_pipeline_and_ablation() -> None:
    parser = build_parser()
    train = parser.parse_args(
        [
            "train", "--init-checkpoint", "models/bc.pt", "--recurrent-size", "256",
            "--rnd-coef", "0.02", "--initial-validation-cursor", "3800",
            "--initial-curriculum-tier", "6",
        ]
    )
    assert train.init_checkpoint == Path("models/bc.pt")
    assert train.recurrent_size == 256
    assert train.rnd_coef == pytest.approx(0.02)
    assert train.initial_validation_cursor == 3800
    assert train.initial_curriculum_tier == 6
    collect = parser.parse_args(
        [
            "imitate", "collect", "--output", "artifacts/data", "--tiers", "1,2",
            "--lesson", "2", "--workers", "8", "--collection-device", "cpu",
            "--success-only",
        ]
    )
    assert collect.imitation_command == "collect"
    assert collect.lesson == 2
    assert collect.workers == 8
    assert collect.collection_device == "cpu"
    assert collect.success_only is True
    evaluation = parser.parse_args(["evaluate", "--seed-start", "3000000", "--episodes", "500"])
    assert evaluation.seed_start == 3_000_000
    direct_collect = build_imitation_parser().parse_args(
        [
            "collect", "--output", "artifacts/data", "--workers", "4",
            "--collection-device", "cpu", "--success-only",
        ]
    )
    assert direct_collect.workers == 4
    assert direct_collect.success_only is True
    direct_bc = build_imitation_parser().parse_args(
        ["bc", "--dataset", "artifacts/data", "--output", "artifacts/bc"]
    )
    assert direct_bc.sequence_length == 64
    assert direct_bc.burn_in == 32
    assert direct_bc.validation_fraction == pytest.approx(0.10)
    assert direct_bc.uniform_window_fraction == pytest.approx(0.50)
    assert direct_bc.validation_windows == 128
    dagger = parser.parse_args(
        [
            "imitate", "dagger", "--base-dataset", "artifacts/teacher",
            "--initial-checkpoint", "artifacts/dagger/round-1/model/best.pt",
            "--output", "artifacts/dagger", "--start-round", "2", "--rounds", "2",
            "--beta-start", "0.5", "--beta-decay", "0.5",
            "--collection-device", "cpu", "--training-device", "cuda",
            "--collection-workers", "12",
        ]
    )
    assert dagger.start_round == 2
    assert dagger.rounds == 2
    assert dagger.beta_start == pytest.approx(0.5)
    assert dagger.beta_decay == pytest.approx(0.5)
    assert dagger.collection_device == "cpu"
    assert dagger.training_device == "cuda"
    assert dagger.collection_workers == 12
    assert dagger.sequence_length == 64
    assert dagger.burn_in == 32
    assert dagger.validation_fraction == pytest.approx(0.10)
    assert dagger.uniform_window_fraction == pytest.approx(0.50)
    assert dagger.validation_windows == 128
    ablate = parser.parse_args(["ablate", "--steps", "1000", "--dry-run"])
    assert ablate.steps == 1000 and ablate.dry_run


def test_checkpoint_selection_requires_validation_and_acceptance_is_consecutive() -> None:
    assert validation_selection_score({}) is None
    assert validation_selection_score({1: 0.96, 2: 0.95, 6: 0.86}) is None
    complete = {1: 0.96, 2: 0.95, 3: 0.97, 4: 0.95, 5: 0.98, 6: 0.86}
    assert validation_selection_score(complete) == pytest.approx((0.86, 0.86))
    curriculum = AdaptiveCurriculum(current_tier=6)
    passing = {1: 0.95, 2: 0.96, 3: 0.97, 4: 0.95, 5: 0.98, 6: 0.86}
    assert next_acceptance_passes(curriculum, passing, 0) == 1
    assert next_acceptance_passes(curriculum, passing, 1) == 2
    failing = {**passing, 6: 0.84}
    assert next_acceptance_passes(curriculum, failing, 1) == 0


def test_seed_namespaces_are_disjoint_and_validation_stays_in_its_50k_block() -> None:
    validation = {
        validation_seed(tier, episode)
        for tier in range(1, 7)
        for episode in (0, 499, 7_999)
    }
    assert min(validation) == 1_000_000
    assert max(validation) == 1_047_999
    assert all(1_000_000 <= seed <= 1_049_999 for seed in validation)
    final = {final_test_seed(3_000_000, tier, 0) for tier in range(1, 7)}
    assert min(final) >= 3_000_000
    assert validation.isdisjoint(final)
