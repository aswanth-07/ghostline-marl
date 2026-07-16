from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio

from ghostline.model import load_policy
from ghostline.policies import FairScriptedPolicy
from ghostline.presentation import GhostlineRenderer
from ghostline.simulation import GhostlineSimulation
from ghostline.types import Action

RECURRENT_POLICY_LABEL = "GRU BC+DAGGER"
RECORDING_SIZE = (1280, 720)
MOVE_LABELS = ("HOLD", "N", "NE", "E", "SE", "S", "SW", "W", "NW")


def record(*, model: Path | None, tier: int, seed: int, output: Path, fps: int = 60) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    sim = GhostlineSimulation(seed=seed, tier=tier)
    renderer = GhostlineRenderer(sim, visible=False)
    torch_policy = load_policy(model) if model else None
    scripted = FairScriptedPolicy()
    observation_env = None
    if torch_policy is not None:
        from ghostline.env import GhostlineEnv

        observation_env = GhostlineEnv(seed=seed, tier=tier)
        observation_env.sim = sim
    hidden = None
    action_value = 0
    action_text = "HOLD"
    policy_label = RECURRENT_POLICY_LABEL if model else "SCRIPTED BASELINE"
    # Capture the same 1280x720 presentation shipped by the browser build.
    # World art remains a clean 2x scale while text is rasterized directly at
    # output resolution instead of enlarging the 640x360 glyph pixels.
    with imageio.get_writer(output, fps=fps, codec="libx264", quality=8, macro_block_size=2) as writer:
        while not (sim.terminated or sim.truncated):
            if sim.elapsed_ticks % 6 == 0:
                if torch_policy is None:
                    action_value = scripted.act(sim)
                else:
                    action_value, hidden = torch_policy.act(observation_env._observation(), hidden, deterministic=True)
            sim.advance(Action.decode(action_value), ticks=1)
            renderer.ingest_events(sim.pop_events())
            decoded = Action.decode(action_value)
            action_text = MOVE_LABELS[decoded.move]
            if decoded.dash:
                action_text += " +DASH"
            if decoded.pulse:
                action_text += " +PULSE"
            frame = renderer.draw(
                return_array=True,
                output_size=RECORDING_SIZE,
                lab_stats={"policy": policy_label, "action": action_text, "latency_ms": 0.0},
            )
            writer.append_data(frame)
        for _ in range(fps * 2):
            writer.append_data(
                renderer.draw(
                    return_array=True,
                    output_size=RECORDING_SIZE,
                    lab_stats={"policy": policy_label, "action": action_text, "latency_ms": 0.0},
                )
            )
    renderer.close()
    if observation_env is not None:
        observation_env.close()
    return output
