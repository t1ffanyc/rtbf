#!/usr/bin/env python3
"""Bar chart comparing Val CER and Test CER across models."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


# Edit this dict to change the comparison data (order = left to right)
DEFAULT_DATA = {
    "Baseline CNN": (20.5, 22.3),
    "Pure RNN": (15.928, 16.274),
    "CNN + RNN Hybrid": (16.216, 19.23),
    "CNN + Transformer": (17.7, 24.85),
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot model comparison (Val CER vs Test CER)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("model_comparison.png"),
        help="Path to save the figure",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Model comparison: Val CER vs Test CER",
        help="Title for the plot",
    )
    args = parser.parse_args()

    models = list(DEFAULT_DATA.keys())
    val_cers = [DEFAULT_DATA[m][0] for m in models]
    test_cers = [DEFAULT_DATA[m][1] for m in models]

    x = range(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar([i - width / 2 for i in x], val_cers, width, label="Val CER", color="#1f77b4")
    bars2 = ax.bar([i + width / 2 for i in x], test_cers, width, label="Test CER", color="#ff7f0e", alpha=0.9)

    ax.set_ylabel("CER (%)")
    ax.set_title(args.title)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    for bars in (bars1, bars2):
        for bar in bars:
            ax.annotate(
                f"{bar.get_height():.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
