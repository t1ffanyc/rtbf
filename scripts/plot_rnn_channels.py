#!/usr/bin/env python3
"""Line chart showing CER vs number of channels for an RNN."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt


# Edit this dict to change the data
# Format: channels: CER
DEFAULT_DATA = {
    8: 33.6,
    12: 35.6,
    14: 34.7,
    16: 16.3,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot channels vs CER for RNN"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("rnn_channels_vs_cer.png"),
        help="Path to save the figure",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="RNN Performance: Channels vs CER",
        help="Title for the plot",
    )
    args = parser.parse_args()

    channels = sorted(DEFAULT_DATA.keys())
    cers = [DEFAULT_DATA[c] for c in channels]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        channels,
        cers,
        marker="o",
        linewidth=2,
    )

    ax.set_xlabel("Number of Channels")
    ax.set_ylabel("CER (%)")
    ax.set_title(args.title)
    ax.set_xticks(channels)
    ax.grid(True, alpha=0.3)

    # Annotate points
    for ch, cer in zip(channels, cers):
        ax.annotate(
            f"{cer:.1f}",
            xy=(ch, cer),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(args.output, dpi=150)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()