#!/usr/bin/env python3
"""Plot epoch vs loss from PyTorch Lightning training log(s)."""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def parse_log(log_path: Path) -> tuple[list[int], list[float]]:
    """Extract (epoch, loss) pairs from log file.

    Expects lines like:
        Epoch 3: 100% 127/127 [00:43<00:00,  2.92it/s, loss=3.37, v_num=0]
    """
    epoch_loss: dict[int, float] = {}
    epoch_re = re.compile(r"Epoch\s+(\d+).*?loss=([\d.]+)")

    text = log_path.read_text()
    for m in epoch_re.finditer(text):
        epoch = int(m.group(1))
        loss = float(m.group(2))
        epoch_loss[epoch] = loss  # keep last occurrence per epoch (100% line)

    if not epoch_loss:
        return [], []

    epochs = sorted(epoch_loss.keys())
    losses = [epoch_loss[e] for e in epochs]
    return epochs, losses


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot epoch vs loss from one or more training logs"
    )
    parser.add_argument(
        "log_paths",
        type=Path,
        nargs="+",
        help="Path(s) to training log file(s)",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default=None,
        help="Comma-separated labels for each log (default: filename stem)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Path to save the figure (default: plot_training_loss.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot (useful in notebooks)",
    )
    args = parser.parse_args()

    log_paths = args.log_paths
    for p in log_paths:
        if not p.exists():
            print(f"Error: {p} does not exist", file=sys.stderr)
            sys.exit(1)

    if args.labels:
        labels = [s.strip() for s in args.labels.split(",")]
        if len(labels) != len(log_paths):
            print(
                f"Error: --labels has {len(labels)} values but {len(log_paths)} logs",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        labels = [p.stem for p in log_paths]

    skip_first = 5
    fig, ax = plt.subplots(figsize=(8, 5))
    for log_path, label in zip(log_paths, labels):
        epochs, losses = parse_log(log_path)
        if epochs:
            mask = [e >= skip_first for e in epochs]
            epochs = [e for e, m in zip(epochs, mask) if m]
            losses = [l for l, m in zip(losses, mask) if m]
        if epochs:
            ax.plot(epochs, losses, "o-", markersize=4, label=label)
        else:
            print(f"Warning: no epoch/loss pairs in {log_path}", file=sys.stderr)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.set_title("Epoch vs loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = args.output
    if out_path is None:
        out_path = Path("plot_training_loss.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
