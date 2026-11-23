from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _latest_metrics_csv(run_dir: Path) -> Path:
    candidates = sorted(run_dir.glob("lightning_logs/version_*/metrics.csv"))
    if not candidates:
        raise FileNotFoundError(f"No metrics.csv found under {run_dir}/lightning_logs")
    return candidates[-1]


def _epoch_table(metrics_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(metrics_csv)
    if "epoch" not in df.columns:
        raise ValueError(f"'epoch' column not found in {metrics_csv}")

    df = df.dropna(subset=["epoch"]).copy()
    df["epoch"] = df["epoch"].astype(int)

    metric_cols = [
        col
        for col in [
            "train_loss_epoch",
            "train_loss",
            "val_loss",
            "val_MAE",
            "val_RMSE",
            "val_MAPE",
            "val_SMAPE",
            "val_mae",
            "val_rmse",
        ]
        if col in df.columns
    ]
    if not metric_cols:
        raise ValueError(f"No plottable metric columns found in {metrics_csv}")

    # Lightning writes several rows per epoch; each metric appears on the row where it was logged.
    return df.groupby("epoch", as_index=False)[metric_cols].last()


def plot_epoch_metrics(run_dir: Path, output: Path | None = None) -> Path:
    metrics_csv = _latest_metrics_csv(run_dir)
    epoch_df = _epoch_table(metrics_csv)

    output = output or run_dir / "figures" / "epoch_metrics.png"
    output.parent.mkdir(parents=True, exist_ok=True)

    loss_cols = [col for col in ["train_loss_epoch", "train_loss", "val_loss"] if col in epoch_df.columns]
    other_cols = [
        col
        for col in ["val_MAE", "val_RMSE", "val_MAPE", "val_SMAPE", "val_mae", "val_rmse"]
        if col in epoch_df.columns
    ]

    n_panels = 1 + bool(other_cols)
    fig, axes = plt.subplots(n_panels, 1, figsize=(10, 4 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    for col in loss_cols:
        axes[0].plot(epoch_df["epoch"], epoch_df[col], marker="o", linewidth=1.8, label=col)
    axes[0].set_title("Loss by Epoch")
    axes[0].set_ylabel("loss")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    if other_cols:
        for col in other_cols:
            axes[1].plot(epoch_df["epoch"], epoch_df[col], marker="o", linewidth=1.5, label=col)
        axes[1].set_title("Validation Metrics by Epoch")
        axes[1].set_ylabel("metric")
        axes[1].grid(True, alpha=0.25)
        axes[1].legend()

    axes[-1].set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    output = plot_epoch_metrics(args.run_dir, args.output)
    print(f"SAVED: {output}")


if __name__ == "__main__":
    main()
