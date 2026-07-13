"""Versioned procedural seed namespaces shared by training and evaluation."""
from __future__ import annotations

TRAINING_SEED_START = 0
TRAINING_SEED_END = 999_999
VALIDATION_SEED_START = 1_000_000
VALIDATION_SEED_END = 1_049_999
VALIDATION_TIER_STRIDE = 8_000
FINAL_TEST_SEED_START = 2_000_000
FINAL_TIER_STRIDE = 100_000


def validation_seed(tier: int, episode: int) -> int:
    if tier not in range(1, 7):
        raise ValueError("tier must lie in 1..6")
    if not 0 <= episode < VALIDATION_TIER_STRIDE:
        raise ValueError(f"validation episode must lie in 0..{VALIDATION_TIER_STRIDE - 1}")
    seed = VALIDATION_SEED_START + (tier - 1) * VALIDATION_TIER_STRIDE + episode
    if seed > VALIDATION_SEED_END:
        raise ValueError("validation seed schedule left its reserved namespace")
    return seed


def final_test_seed(seed_start: int, tier: int, episode: int) -> int:
    if seed_start < FINAL_TEST_SEED_START:
        raise ValueError(f"final seed_start must be at least {FINAL_TEST_SEED_START:,}")
    if tier not in range(1, 7):
        raise ValueError("tier must lie in 1..6")
    if episode < 0:
        raise ValueError("episode must be non-negative")
    return int(seed_start) + tier * FINAL_TIER_STRIDE + int(episode)
