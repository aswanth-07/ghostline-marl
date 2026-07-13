from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio

from ghostline.model import load_policy
from ghostline.policies import FairScriptedPolicy
from ghostline.presentation import GhostlineRenderer
from ghostline.simulation import GhostlineSimulation
from ghostline.types import Action


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
    with imageio.get_writer(output, fps=fps, codec="libx264", quality=8, macro_block_size=16) as writer:
        while not (sim.terminated or sim.truncated):
            if sim.elapsed_ticks % 6 == 0:
                if torch_policy is None:
                    action_value = scripted.act(sim)
                else:
                    action_value, hidden = torch_policy.act(observation_env._observation(), hidden, deterministic=True)
            sim.advance(Action.decode(action_value), ticks=1)
            renderer.ingest_events(sim.pop_events())
            frame = renderer.draw(return_array=True, lab_stats={"policy": "RECURRENT PPO" if model else "SCRIPTED BASELINE"})
            writer.append_data(frame)
        for _ in range(fps * 2):
            writer.append_data(renderer.draw(return_array=True, lab_stats={"policy": "RECURRENT PPO" if model else "SCRIPTED BASELINE"}))
    renderer.close()
    if observation_env is not None:
        observation_env.close()
    return output
