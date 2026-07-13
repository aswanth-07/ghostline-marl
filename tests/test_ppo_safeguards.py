from __future__ import annotations

import numpy as np
import pytest
import torch
from argparse import Namespace
from pathlib import Path

from ghostline.model import UniversalGhostlinePolicy, save_policy
from ghostline.training import _training_args

from ghostline.torchrl_train import (
    _load_initial_policy,
    _prepare_initial_rollback,
    _require_matching_resume_contract,
    categorical_anchor_kl,
    mask_terminal_intrinsic_rewards,
    require_validation_window,
    validate_curriculum,
    validation_selection_score,
)


def test_same_step_terminal_reset_novelty_is_not_rewarded() -> None:
    intrinsic = torch.tensor((0.5, 2.0, 1.25))
    masked = mask_terminal_intrinsic_rewards(
        intrinsic, np.asarray((False, True, False))
    )
    assert masked.tolist() == pytest.approx((0.5, 0.0, 1.25))
    with pytest.raises(ValueError, match="must match"):
        mask_terminal_intrinsic_rewards(intrinsic, np.asarray((True, False)))


def test_policy_anchor_kl_is_zero_for_identity_and_positive_for_drift() -> None:
    reference = torch.tensor(((2.0, 0.0, -1e9), (0.0, 1.0, 2.0)))
    assert torch.max(categorical_anchor_kl(reference, reference).abs()) < 1e-7
    drifted = reference.clone()
    drifted[:, :2] = drifted[:, :2].flip(-1)
    assert torch.all(categorical_anchor_kl(drifted, reference) > 0.0)
    with pytest.raises(ValueError, match="identical shapes"):
        categorical_anchor_kl(drifted[:, :2], reference)


def test_curriculum_validation_uses_requested_disjoint_offset(monkeypatch) -> None:
    seen: list[tuple[int, int]] = []

    def fake_validate(_policy, tier, episodes, _device, *, validation_offset=0):
        seen.append((tier, validation_offset))
        return episodes / 10.0

    monkeypatch.setattr("ghostline.torchrl_train.validate", fake_validate)
    rates = validate_curriculum(
        object(),
        current_tier=3,
        episodes=2,
        device=torch.device("cpu"),
        validation_offset=47,
    )
    assert rates == {1: 0.2, 2: 0.2, 3: 0.2}
    assert seen == [(1, 47), (2, 47), (3, 47)]


def test_ppo_initialization_checkpoint_is_returned_in_training_mode(tmp_path: Path) -> None:
    checkpoint = tmp_path / "initial.pt"
    save_policy(UniversalGhostlinePolicy(recurrent_size=256), checkpoint)
    args = Namespace(
        feedforward=False,
        init_checkpoint=checkpoint,
        recurrent_size=256,
    )
    policy = _load_initial_policy(args, torch.device("cpu"))
    assert policy.training is True
    assert policy.core is not None and policy.core.training is True


def test_checkpoint_selection_requires_complete_six_tier_coverage() -> None:
    assert validation_selection_score({1: 1.0}) is None
    assert validation_selection_score({1: 0.96, 2: 0.95, 6: 0.86}) is None
    rates = {1: 0.96, 2: 0.95, 3: 0.97, 4: 0.98, 5: 0.95, 6: 0.86}
    assert validation_selection_score(rates) == pytest.approx((0.86, 0.86))


def test_initial_rollback_is_created_once_and_source_bound(tmp_path: Path) -> None:
    source = tmp_path / "dagger.pt"
    rollback = tmp_path / "initial-rollback.pt"
    policy = UniversalGhostlinePolicy(recurrent_size=256)
    save_policy(policy, source)
    digest, anchored = _prepare_initial_rollback(
        policy,
        rollback,
        source=source,
        allow_existing=False,
    )
    original = rollback.read_bytes()
    assert anchored is True and len(digest) == 64

    resumed_digest, resumed_anchored = _prepare_initial_rollback(
        policy,
        rollback,
        source=source,
        allow_existing=True,
    )
    assert resumed_digest == digest and resumed_anchored is True
    assert rollback.read_bytes() == original

    save_policy(UniversalGhostlinePolicy(recurrent_size=256), source, changed=True)
    with pytest.raises(RuntimeError, match="differs from the immutable"):
        _prepare_initial_rollback(
            policy,
            rollback,
            source=source,
            allow_existing=True,
        )


def test_resume_contract_and_validation_cursor_fail_closed() -> None:
    expected = {"initial_validation_cursor": 3800, "initial_curriculum_tier": 6}
    _require_matching_resume_contract({"resume_contract": dict(expected)}, expected)
    with pytest.raises(RuntimeError, match="resume contract changed"):
        _require_matching_resume_contract(
            {"resume_contract": {**expected, "initial_validation_cursor": 4000}},
            expected,
        )
    with pytest.raises(RuntimeError, match="predates"):
        _require_matching_resume_contract({}, expected)

    require_validation_window(3800, 50)
    require_validation_window(7950, 50)
    with pytest.raises(ValueError, match="positive"):
        require_validation_window(0, 0)
    with pytest.raises(ValueError, match="namespace"):
        require_validation_window(7951, 50)


def test_public_training_launcher_forwards_safe_start_controls() -> None:
    command = _training_args(
        hours=1.0,
        experiment="safe-ppo",
        initial_validation_cursor=3800,
        initial_curriculum_tier=6,
    )
    assert "--initial-validation-cursor=3800" in command
    assert "--initial-curriculum-tier=6" in command
