#!/usr/bin/env python3

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def parse_log(log_path: Path) -> tuple[list[int], list[float]]:

    epoch_loss: dict[int, float] = {}
    epoch_re = re.compile(r"Epoch\s+(\d+).*?loss=([\d.]+)")
    
    text = log_path.read_text()
    for m in epoch_re.finditer(text):
        epoch = int(m.group(1))
        loss = float(m.group(2))
        epoch_loss[epoch] = loss
    
    if not epoch_loss:
        return [], []
    
    epochs = sorted(epoch_loss.keys())
    losses = [epoch_loss[e] for e in epochs]
    return epochs, losses


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot epoch vs loss from training log")
    parser.add_argument(
        "log_path",
        type=Path,
        help="Path to the training log file",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Path to save the figure (default: plot_training_loss.png in log dir)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot (useful in notebooks)",
    )
    args = parser.parse_args()

    log_path = args.log_path
    if not log_path.exists():
        print(f"Error: {log_path} does not exist", file=sys.stderr)
        sys.exit(1)

    epochs, losses = parse_log(log_path)
    if not epochs:
        print("No epoch/loss pairs found in log.", file=sys.stderr)
        sys.exit(1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, losses, "o-", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title("Epoch vs loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = args.output
    if out_path is None:
        out_path = log_path.parent / "plot_training_loss.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
