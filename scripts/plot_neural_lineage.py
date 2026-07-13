from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "benchmarks" / "neural" / "training-lineage.csv"
OUTPUT = ROOT / "assets" / "screenshots" / "neural-training-lineage.png"


def main() -> None:
    with INPUT.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    labels = [row["stage"] for row in rows]
    x = list(range(len(labels)))
    colors = ("#62e6ff", "#f4c95d", "#7bd389", "#dd7cff", "#ff8f70", "#f25f5c")

    plt.style.use("dark_background")
    figure, axis = plt.subplots(figsize=(13.2, 6.5), dpi=150)
    figure.patch.set_facecolor("#080d14")
    axis.set_facecolor("#0d1620")
    for tier, color in zip(range(1, 7), colors, strict=True):
        values = [100.0 * float(row[f"tier_{tier}"]) for row in rows]
        axis.plot(x, values, color=color, marker="o", linewidth=2.0, label=f"Tier {tier}")

    axis.axhline(95, color="#b8c4cf", linestyle="--", linewidth=1.0, alpha=0.55)
    axis.axhline(85, color="#f25f5c", linestyle=":", linewidth=1.0, alpha=0.65)
    axis.axvline(4.5, color="#62e6ff", linestyle="--", linewidth=1.0, alpha=0.35)
    axis.text(4.55, 32, "checkpoint frozen after targeted DAgger", color="#9defff", fontsize=8)
    axis.set_ylim(30, 102)
    axis.set_ylabel("Deterministic success rate (%)")
    axis.set_title("Ghostline neural policy: closed-loop generalization lineage", loc="left", pad=14)
    axis.set_xticks(x, labels, rotation=24, ha="right")
    axis.grid(axis="y", color="#718096", alpha=0.18)
    axis.legend(ncol=3, frameon=False, loc="lower right")
    figure.text(
        0.01,
        0.01,
        "BC architectures share offset 3000; DAgger rounds use disjoint validation windows; final uses 500 untouched 7M seeds/tier.",
        color="#9aa7b3",
        fontsize=8,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, bbox_inches="tight", facecolor=figure.get_facecolor())
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
