from __future__ import annotations

from pathlib import Path
import time
from typing import Mapping

import numpy as np


class OnnxGhostlinePolicy:
    """Small player-runtime adapter; intentionally has no PyTorch dependency."""

    def __init__(self, path: Path):
        import onnxruntime as ort

        self.path = Path(path)
        self.session = ort.InferenceSession(str(self.path), providers=["CPUExecutionProvider"])
        self.input_names = {item.name for item in self.session.get_inputs()}
        hidden_input = next(item for item in self.session.get_inputs() if item.name == "hidden")
        hidden_size = hidden_input.shape[-1]
        if not isinstance(hidden_size, int):
            raise ValueError("ONNX policy must expose a static recurrent hidden width")
        self.hidden_size = hidden_size
        self.last_latency_ms = 0.0

    def act(
        self,
        observation: Mapping[str, np.ndarray],
        hidden: np.ndarray | None = None,
        *,
        deterministic: bool = True,
        **_: object,
    ) -> tuple[int, np.ndarray]:
        if hidden is None:
            hidden = np.zeros((1, 1, self.hidden_size), dtype=np.float32)
        feed = {
            key: np.expand_dims(np.asarray(value), 0)
            for key, value in observation.items()
            if key in self.input_names
        }
        feed["hidden"] = np.asarray(hidden, dtype=np.float32)
        started = time.perf_counter()
        logits, _, next_hidden = self.session.run(None, feed)
        self.last_latency_ms = (time.perf_counter() - started) * 1000.0
        if deterministic:
            action = int(np.argmax(logits[0]))
        else:
            probabilities = np.exp(logits[0] - np.max(logits[0]))
            probabilities /= probabilities.sum()
            action = int(np.random.default_rng().choice(len(probabilities), p=probabilities))
        return action, np.asarray(next_hidden, dtype=np.float32)
