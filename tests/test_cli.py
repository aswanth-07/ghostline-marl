from __future__ import annotations

from pathlib import Path

from neon_arena.config import MAX_STEPS
from neon_arena.cli import build_parser


def test_record_command_defaults_to_video_output() -> None:
    args = build_parser().parse_args(["record", "--random"])

    assert args.command == "record"
    assert args.random is True
    assert args.output == Path("videos/core-runner-recording.mp4")
    assert args.steps == MAX_STEPS
    assert args.fps == 60


def test_train_record_pipeline_defaults_to_sparse_video() -> None:
    args = build_parser().parse_args(["train-record"])

    assert args.command == "train-record"
    assert args.n_envs == 16
    assert args.episodes is None
    assert args.record_every_episodes == 100
    assert args.record_steps == 480
    assert args.record_full_episode is False
    assert args.record_fps == 30
    assert args.video == Path("videos/training-progress-v6.mp4")


def test_episode_training_resume_arguments() -> None:
    args = build_parser().parse_args(
        [
            "train-record",
            "--episodes",
            "10000",
            "--resume-from",
            "models/blackline-heist-v3",
        ]
    )

    assert args.episodes == 10_000
    assert args.resume_from == Path("models/blackline-heist-v3")
    assert args.no_progress is False


def test_full_episode_recording_flag_is_available() -> None:
    args = build_parser().parse_args(["train-record", "--record-full-episode"])

    assert args.record_full_episode is True
    assert args.record_steps == 480
    assert MAX_STEPS == 2500


def test_play_command_curriculum_argument() -> None:
    args = build_parser().parse_args(["play"])
    assert args.command == "play"
    assert args.curriculum == "full"

    args_custom = build_parser().parse_args(["play", "--curriculum", "large_easy"])
    assert args_custom.curriculum == "large_easy"
