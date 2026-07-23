from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

from scripts import verify_security_release_evidence as security_evidence


ROOT = Path(__file__).resolve().parents[1]


def _copy_security_release_tree(destination: Path) -> None:
    model = destination / security_evidence.SECURITY_CHECKPOINT
    model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / security_evidence.SECURITY_CHECKPOINT, model)
    reports = (
        security_evidence.FINAL_REPORT,
        security_evidence.FAILED_FINAL_REPORT,
        *security_evidence.VALIDATION_REPORTS,
        security_evidence.TEACHER_REPORT,
    )
    for relative in reports:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
        shutil.copy2((ROOT / relative).with_suffix(".csv"), target.with_suffix(".csv"))
        shutil.copy2(
            (ROOT / relative).with_name(f"{relative.stem}.episodes.csv"),
            target.with_name(f"{relative.stem}.episodes.csv"),
        )


def test_security_release_evidence_binds_checkpoint_and_seed_slices(tmp_path: Path) -> None:
    _copy_security_release_tree(tmp_path)

    summary = security_evidence.verify_security_release(tmp_path)

    assert summary["status"] == "passed"
    assert summary["final_episodes"] == 100
    assert summary["mean_stop_rate"] == pytest.approx(0.07)
    assert summary["tier_stop_rates"] == {"3": 0.04, "4": 0.0, "5": 0.08, "6": 0.16}


def test_security_release_evidence_rejects_report_checkpoint_mismatch(tmp_path: Path) -> None:
    _copy_security_release_tree(tmp_path)
    report_path = tmp_path / security_evidence.FINAL_REPORT
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["security_checkpoint_sha256"] = "0" * 64
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(security_evidence.SecurityReleaseEvidenceError, match="checkpoint hash"):
        security_evidence.verify_security_release(tmp_path)
