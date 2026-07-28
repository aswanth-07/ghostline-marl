from __future__ import annotations

import copy
import os

import pytest

from ghostline.config_v2 import HACK_DEVICES_PER_TIER, VENT_PAIRS_PER_TIER
from ghostline.generation_v2 import (
    MAX_RESHAPE_ATTEMPTS,
    FacilityLayoutV2,
)
from ghostline.types import Prop


def _content_signature(level) -> tuple:
    return (
        level.grid.tobytes(),
        tuple(
            (prop.kind, prop.tile_x, prop.tile_y, prop.width, prop.height, prop.blocking)
            for prop in level.props
        ),
        tuple(
            (
                vent.vent_id,
                vent.tile,
                vent.exit_tile,
                tuple(float(value) for value in vent.exit_position),
            )
            for vent in level.vents
        ),
        tuple(
            (
                device.device_id,
                device.kind,
                device.tile,
                device.target_id,
                device.target_tile,
            )
            for device in level.hackable
        ),
    )


@pytest.mark.parametrize("tier", range(1, 7))
def test_v2_generation_is_deterministic_and_ready(tier: int) -> None:
    seed = 4_850_000 + tier
    generator = FacilityLayoutV2()
    first = generator.generate(seed=seed, tier=tier)
    second = generator.generate(seed=seed, tier=tier)

    assert _content_signature(first) == _content_signature(second)
    assert generator.readiness_errors(first) == ()
    assert len(first.vents) == VENT_PAIRS_PER_TIER[tier] * 2
    assert len(first.hackable) == HACK_DEVICES_PER_TIER[tier]


def test_v2_readiness_validator_rejects_visual_overlap() -> None:
    generator = FacilityLayoutV2()
    level = generator.generate(seed=4_850_101, tier=6)
    broken = copy.deepcopy(level)
    occupied = broken.vents[0].tile
    broken.props.append(Prop("floor_marking", occupied[0], occupied[1], 1, 1, False))

    errors = generator.readiness_errors(broken)
    assert "prop_overlap" in errors
    assert not generator.validate(broken)


def test_v2_readiness_validator_rejects_useless_door_panel() -> None:
    generator = FacilityLayoutV2()
    level = generator.generate(seed=4_850_102, tier=3)
    broken = copy.deepcopy(level)
    device = broken.hackable[0]
    device.kind = "door"
    device.target_id = 0
    device.target_tile = broken.doors[0].tile

    errors = generator.readiness_errors(broken)
    assert "useless_door_panel" in errors
    assert "door_target" in errors


def test_v2_generation_failure_is_bounded_and_deterministic() -> None:
    class RejectingLayout(FacilityLayoutV2):
        def __init__(self) -> None:
            self.attempts = 0

        def _reshape(self, base, rng):  # noqa: ANN001
            self.attempts += 1
            return base

        def validate(self, level) -> bool:  # noqa: ANN001
            return False

    first = RejectingLayout()
    second = RejectingLayout()
    message = (
        "could not generate a valid Env-v2 tier 6 level for seed 4850199 "
        f"after {MAX_RESHAPE_ATTEMPTS} deterministic attempts"
    )
    with pytest.raises(RuntimeError, match="could not generate") as first_error:
        first.generate(seed=4_850_199, tier=6)
    with pytest.raises(RuntimeError, match="could not generate") as second_error:
        second.generate(seed=4_850_199, tier=6)

    assert str(first_error.value) == message
    assert str(second_error.value) == message
    assert first.attempts == second.attempts == MAX_RESHAPE_ATTEMPTS


def test_v2_generation_fuzz_gate() -> None:
    """Fast in CI; set GHOSTLINE_V2_FUZZ_SEEDS=10000 for the release gate."""

    count = int(os.environ.get("GHOSTLINE_V2_FUZZ_SEEDS", "72"))
    assert count > 0
    generator = FacilityLayoutV2()
    failures: list[tuple[int, int, tuple[str, ...]]] = []
    for index in range(count):
        seed = 4_900_000 + index
        tier = index % 6 + 1
        level = generator.generate(seed=seed, tier=tier)
        errors = generator.readiness_errors(level)
        if errors:
            failures.append((seed, tier, errors))
        if tier <= 3:
            assert all(device.kind != "door" for device in level.hackable)
        selected_door_tiles = {
            level.doors[index].tile for index in generator._lockable_door_indices(level)
        }
        assert all(
            device.target_tile in selected_door_tiles
            for device in level.hackable
            if device.kind == "door"
        )
        # Readiness validation is meant to cover all active placements, not
        # merely classic terminal/extraction reachability.
        assert len({vent.tile for vent in level.vents}) == len(level.vents)
        assert not ({vent.tile for vent in level.vents} & {device.tile for device in level.hackable})

    assert failures == []
