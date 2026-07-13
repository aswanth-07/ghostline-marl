from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


PROMOTION_THRESHOLDS = {1: 0.90, 2: 0.90, 3: 0.86, 4: 0.82, 5: 0.78, 6: 0.85}
ACCEPTANCE_THRESHOLDS = {1: 0.95, 2: 0.95, 3: 0.95, 4: 0.95, 5: 0.95, 6: 0.85}


@dataclass
class AdaptiveCurriculum:
    current_tier: int = 1
    consecutive_passes: int = 0
    validation_history: dict[int, list[float]] = field(default_factory=lambda: {tier: [] for tier in range(1, 7)})

    def observe_validation(self, tier: int, success_rate: float) -> bool:
        self.validation_history[tier].append(float(success_rate))
        if tier != self.current_tier:
            return False
        if success_rate >= PROMOTION_THRESHOLDS[tier]:
            self.consecutive_passes += 1
        else:
            self.consecutive_passes = 0
        if self.consecutive_passes >= 2 and self.current_tier < 6:
            self.current_tier += 1
            self.consecutive_passes = 0
            return True
        return False

    def sample_tier(self, rng: np.random.Generator) -> int:
        if self.current_tier == 1:
            return 1
        if self.current_tier < 6:
            if rng.random() < 0.70:
                return self.current_tier
            return int(rng.integers(1, self.current_tier))
        if rng.random() < 0.50:
            return 6
        return int(rng.integers(1, 6))

    def acceptance_met(self, rates: dict[int, float]) -> bool:
        return all(rates.get(tier, 0.0) >= threshold for tier, threshold in ACCEPTANCE_THRESHOLDS.items())
