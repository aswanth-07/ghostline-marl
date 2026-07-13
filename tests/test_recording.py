from __future__ import annotations

from pathlib import Path

import numpy as np


def test_recording_preserves_logical_canvas_without_macroblock_rescale(
    tmp_path: Path, monkeypatch
) -> None:
    import ghostline.recording as recording

    writer_options: dict[str, object] = {}
    written_shapes: list[tuple[int, ...]] = []
    labels: list[str] = []

    class FinishedSimulation:
        terminated = True
        truncated = False

        def __init__(self, **_kwargs) -> None:
            pass

    class Renderer:
        def __init__(self, _sim, *, visible: bool) -> None:
            assert not visible

        def draw(self, *, return_array: bool, lab_stats: dict[str, str]):
            assert return_array
            labels.append(lab_stats["policy"])
            return np.zeros((360, 640, 3), dtype=np.uint8)

        def close(self) -> None:
            pass

    class Writer:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            pass

        def append_data(self, frame: np.ndarray) -> None:
            written_shapes.append(frame.shape)

    def get_writer(_output: Path, **kwargs):
        writer_options.update(kwargs)
        return Writer()

    monkeypatch.setattr(recording, "GhostlineSimulation", FinishedSimulation)
    monkeypatch.setattr(recording, "GhostlineRenderer", Renderer)
    monkeypatch.setattr(recording.imageio, "get_writer", get_writer)

    recording.record(model=None, tier=1, seed=0, output=tmp_path / "demo.mp4", fps=1)

    assert writer_options["macro_block_size"] == 2
    assert written_shapes == [(360, 640, 3), (360, 640, 3)]
    assert labels == ["SCRIPTED BASELINE", "SCRIPTED BASELINE"]
    assert recording.RECURRENT_POLICY_LABEL == "GRU BC+DAGGER"
