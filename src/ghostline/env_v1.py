"""Published single-agent Ghostline contract.

The released simulation, observation, action, checkpoint, and evaluation bytes
historically called themselves ``GhostlineEnv-v2``.  Those frozen source files
must remain byte-identical because the published model and its 3,000-contract
audit are bound to their source fingerprint.  This zero-mechanics wrapper gives
that exact released environment its final public name, ``GhostlineEnv-v1``,
without rewriting or weakening any historical evidence.
"""

from __future__ import annotations

from typing import Any

from ghostline.env import GhostlineEnv


PUBLISHED_ENVIRONMENT_ID = "GhostlineEnv-v1"
HISTORICAL_INTERNAL_CONTRACT = "GhostlineEnv-v2"


class PublishedGhostlineEnvV1(GhostlineEnv):
    """Exact published single-agent game exposed under its stable public ID."""

    def _info(self) -> dict[str, Any]:
        info = super()._info()
        info["contract"] = PUBLISHED_ENVIRONMENT_ID
        info["historical_internal_contract"] = HISTORICAL_INTERNAL_CONTRACT
        return info

    def telemetry(self) -> dict[str, Any]:
        telemetry = super().telemetry()
        telemetry["contract"] = PUBLISHED_ENVIRONMENT_ID
        telemetry["historical_internal_contract"] = HISTORICAL_INTERNAL_CONTRACT
        return telemetry
