#!/usr/bin/env python3
"""Plot training loss and val/CER over epochs from a PyTorch Lightning log."""

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def parse_log(log_path: Path) -> tuple[list[int], list[float], list[float]]:
    """Extract (epochs, losses, val_cers) from log file.

    - Loss from: Epoch N: 100% ... loss=X
    - val/CER from: 'val/CER' reached X or 'val/CER' was not in top 1 (forward-fill)
    """
    text = log_path.read_text()

    # Epoch -> loss (keep last per epoch, i.e. 100% line)
    epoch_loss: dict[int, float] = {}
    for m in re.finditer(r"Epoch\s+(\d+).*?loss=([\d.]+)", text):
        epoch_loss[int(m.group(1))] = float(m.group(2))

    # Epoch -> val CER (process in order; forward-fill on "was not in top 1")
    epoch_cer: dict[int, float] = {}
    last_cer: float | None = None

    pattern = re.compile(
        r"Epoch\s+(\d+),.*?'val/CER'\s+(?:reached\s+([\d.]+)|was not in top 1)"
    )
    for m in pattern.finditer(text):
        epoch = int(m.group(1))
        if m.group(2) is not None:
            last_cer = float(m.group(2))
        if last_cer is not None:
            epoch_cer[epoch] = last_cer

    epochs = sorted(set(epoch_loss) & set(epoch_cer))
    losses = [epoch_loss[e] for e in epochs]
    cers = [epoch_cer[e] for e in epochs]
    return epochs, losses, cers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot loss and val/CER vs epoch from a training log"
    )
    parser.add_argument(
        "log_path",
        type=Path,
        help="Path to the training log file",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Training loss and val/CER vs epoch",
        metavar="TEXT",
        help="Title for the plot (use --title=\"My Title\" for spaces)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Path to save the figure (default: plot_loss_vs_cer.png in log dir)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the plot",
    )
    args = parser.parse_args()

    log_path = args.log_path
    if not log_path.exists():
        print(f"Error: {log_path} does not exist", file=sys.stderr)
        sys.exit(1)

    epochs, losses, cers = parse_log(log_path)
    if not epochs:
        print("No data found in log.", file=sys.stderr)
        sys.exit(1)

    skip_first = 30
    mask = [e >= skip_first for e in epochs]
    epochs = [e for e, m in zip(epochs, mask) if m]
    losses = [l for l, m in zip(losses, mask) if m]
    cers = [c for c, m in zip(cers, mask) if m]
    if not epochs:
        print("No data remaining after skipping first 5 epochs.", file=sys.stderr)
        sys.exit(1)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    ax1.plot(epochs, losses, "o-", markersize=4, color="#1f77b4", label="training loss")
    ax2.plot(epochs, cers, "s--", markersize=4, color="#ff7f0e", alpha=0.9, label="val CER")

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Training loss")
    ax2.set_ylabel("val/CER (%)")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.set_title(args.title)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path = args.output or (log_path.parent / "plot_loss_vs_cer.png")
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")

    if args.show:
        plt.show()
    else:
        plt.close()


if __name__ == "__main__":
    main()
