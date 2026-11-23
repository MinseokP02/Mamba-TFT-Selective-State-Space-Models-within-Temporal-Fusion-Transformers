# scripts/run_all_cnn2.py
from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from pv_forecast.common.io import ensure_dir
from pv_forecast.common.paths import copy_config, make_run_dir, save_json
from pv_forecast.common.seed import seed_all
from pv_forecast.data.build_df_full import build_df_full
from scripts.eval_cnn_impl import eval_cnn_main


def _unique_keep_order(cols: list[str]) -> list[str]:
    return list(dict.fromkeys(cols))


def _inverse_target(values: np.ndarray, scaler: StandardScaler, target_idx: int) -> np.ndarray:
    return values * scaler.scale_[target_idx] + scaler.mean_[target_idx]


def _first_present(mapping: dict[str, Any], keys: list[str], default: Any) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _resolve_split(cfg: dict[str, Any], df_full: pd.DataFrame) -> dict[str, Any]:
    data_cfg = cfg["data"]
    split_cfg = cfg.get("split", {})

    train_ratio = float(split_cfg.get("train_ratio", 0.7))
    val_ratio = float(split_cfg.get("val_ratio", 0.15))
    test_ratio = 1.0 - train_ratio - val_ratio

    if not (0 < train_ratio < 1):
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    if not (0 <= val_ratio < 1):
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    if test_ratio <= 0:
        raise ValueError(
            f"Invalid split ratios: train_ratio({train_ratio}) + "
            f"val_ratio({val_ratio}) must be < 1.0"
        )

    time_idx_col = data_cfg.get("time_idx_col", "time_idx")
    n = len(df_full)
    train_cutoff = int(n * train_ratio)
    val_cutoff = int(n * (train_ratio + val_ratio))

    train_cutoff_time_idx = int(df_full.iloc[train_cutoff][time_idx_col])
    val_cutoff_time_idx = int(df_full.iloc[val_cutoff][time_idx_col])

    return {
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "n_total": n,
        "train_cutoff": train_cutoff,
        "val_cutoff": val_cutoff,
        "train_cutoff_time_idx": train_cutoff_time_idx,
        "val_cutoff_time_idx": val_cutoff_time_idx,
        "training_cutoff": train_cutoff_time_idx,
        "validation_cutoff": val_cutoff_time_idx,
        "test_start_time_idx": val_cutoff_time_idx,
        "test_end_time_idx": int(df_full[time_idx_col].max()),
        "train_rows": train_cutoff,
        "val_rows": val_cutoff - train_cutoff,
        "test_rows": n - val_cutoff,
    }


@dataclass
class CNN2Sample:
    x: np.ndarray
    y: np.ndarray
    group_id: str
    decoder_time_idx_start: int
    decoder_time_idx_end: int


class CNN2KnownFutureDataset(Dataset):
    """
    CNN baseline that is closer to TFT's information setup.

    Encoder part:
      - uses all model_feature_cols, including past target.

    Decoder part:
      - uses known_future_cols only.
      - target and other unknown future channels are set to 0 in scaled space.

    A decoder indicator channel is appended so the CNN can distinguish history
    from future-known slots.
    """

    def __init__(
        self,
        df_full: pd.DataFrame,
        model_feature_cols: list[str],
        known_future_cols: list[str],
        target_col: str,
        group_col: str,
        time_idx_col: str,
        encoder_length: int,
        prediction_length: int,
        split: str,
        training_cutoff: int,
        validation_cutoff: int,
        test_start_time_idx: int | None = None,
        allow_missing_time_idx: bool = False,
    ):
        super().__init__()

        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split={split}")

        self.df_full = df_full.sort_values([group_col, time_idx_col]).reset_index(drop=True)
        self.model_feature_cols = model_feature_cols
        self.known_future_cols = known_future_cols
        self.target_col = target_col
        self.group_col = group_col
        self.time_idx_col = time_idx_col
        self.encoder_length = encoder_length
        self.prediction_length = prediction_length
        self.split = split
        self.training_cutoff = training_cutoff
        self.validation_cutoff = validation_cutoff
        self.test_start_time_idx = test_start_time_idx
        self.allow_missing_time_idx = allow_missing_time_idx

        self.feature_index = {col: i for i, col in enumerate(model_feature_cols)}
        self.known_future_indices = [
            self.feature_index[col]
            for col in known_future_cols
            if col in self.feature_index and col != target_col
        ]

        self.samples: list[CNN2Sample] = []
        self._build_samples()

    def _is_contiguous(self, arr: np.ndarray) -> bool:
        return len(arr) <= 1 or bool(np.all(np.diff(arr) == 1))

    def _match_split(self, decoder_start_t: int, decoder_end_t: int) -> bool:
        if self.split == "train":
            return decoder_end_t < self.training_cutoff
        if self.split == "val":
            return (
                decoder_start_t >= self.training_cutoff
                and decoder_end_t < self.validation_cutoff
            )

        start = self.test_start_time_idx
        if start is None:
            start = self.validation_cutoff
        return decoder_start_t >= start

    def _build_samples(self) -> None:
        total_length = self.encoder_length + self.prediction_length
        n_features = len(self.model_feature_cols)

        for group_id, gdf in self.df_full.groupby(self.group_col):
            gdf = gdf.sort_values(self.time_idx_col).reset_index(drop=True)

            if len(gdf) < total_length:
                continue

            time_idx_arr = gdf[self.time_idx_col].to_numpy()
            feature_arr = gdf[self.model_feature_cols].to_numpy(dtype=np.float32)
            target_arr = gdf[self.target_col].to_numpy(dtype=np.float32)

            for start in range(0, len(gdf) - total_length + 1):
                end = start + total_length
                window_time_idx = time_idx_arr[start:end]

                if not self.allow_missing_time_idx and not self._is_contiguous(window_time_idx):
                    continue

                enc_start = start
                enc_end = start + self.encoder_length
                dec_end = end

                decoder_start_t = int(time_idx_arr[enc_end])
                decoder_end_t = int(time_idx_arr[dec_end - 1])

                if not self._match_split(decoder_start_t, decoder_end_t):
                    continue

                x = np.zeros((total_length, n_features + 1), dtype=np.float32)
                x[: self.encoder_length, :n_features] = feature_arr[enc_start:enc_end]
                x[self.encoder_length :, self.known_future_indices] = feature_arr[
                    enc_end:dec_end,
                    self.known_future_indices,
                ]
                x[self.encoder_length :, -1] = 1.0

                y = target_arr[enc_end:dec_end]

                self.samples.append(
                    CNN2Sample(
                        x=x.T,
                        y=y,
                        group_id=str(group_id),
                        decoder_time_idx_start=decoder_start_t,
                        decoder_time_idx_end=decoder_end_t,
                    )
                )

        if not self.samples:
            raise RuntimeError(
                f"No samples created for split={self.split}. "
                "Check cutoffs, encoder_length, prediction_length, and time_idx continuity."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        meta = {
            "group_id": item.group_id,
            "decoder_time_idx_start": item.decoder_time_idx_start,
            "decoder_time_idx_end": item.decoder_time_idx_end,
        }
        return (
            torch.tensor(item.x, dtype=torch.float32),
            torch.tensor(item.y, dtype=torch.float32),
            meta,
        )


class CNN2KnownFutureDataModule(pl.LightningDataModule):
    def __init__(
        self,
        df_full: pd.DataFrame,
        model_feature_cols: list[str],
        known_future_cols: list[str],
        target_col: str,
        group_col: str,
        time_idx_col: str,
        encoder_length: int,
        prediction_length: int,
        training_cutoff: int,
        validation_cutoff: int,
        test_start_time_idx: int,
        batch_size: int,
        num_workers: int = 0,
        allow_missing_time_idx: bool = False,
    ):
        super().__init__()
        self.kwargs = dict(
            df_full=df_full,
            model_feature_cols=model_feature_cols,
            known_future_cols=known_future_cols,
            target_col=target_col,
            group_col=group_col,
            time_idx_col=time_idx_col,
            encoder_length=encoder_length,
            prediction_length=prediction_length,
            training_cutoff=training_cutoff,
            validation_cutoff=validation_cutoff,
            test_start_time_idx=test_start_time_idx,
            allow_missing_time_idx=allow_missing_time_idx,
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: str | None = None):
        if stage in (None, "fit"):
            self.train_dataset = CNN2KnownFutureDataset(split="train", **self.kwargs)
            self.val_dataset = CNN2KnownFutureDataset(split="val", **self.kwargs)
        if stage in ("test", "predict"):
            self.test_dataset = CNN2KnownFutureDataset(split="test", **self.kwargs)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class CNN2QuantileNet(nn.Module):
    def __init__(
        self,
        num_features: int,
        sequence_length: int,
        output_dim: int,
        quantiles: list[float],
        conv_out_channels: int = 64,
        kernel_size: int = 3,
        hidden_dim1: int = 128,
        hidden_dim2: int = 64,
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        padding = kernel_size // 2
        act = nn.GELU if activation == "gelu" else nn.ReLU

        self.output_dim = output_dim
        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)

        self.conv = nn.Sequential(
            nn.Conv1d(num_features, conv_out_channels, kernel_size, padding=padding),
            act(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_out_channels, conv_out_channels, kernel_size, padding=padding),
            act(),
            nn.Dropout(dropout),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_channels * sequence_length, hidden_dim1),
            act(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            act(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, output_dim * self.num_quantiles),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        y_hat = self.head(x)
        return y_hat.view(x.size(0), self.output_dim, self.num_quantiles)


class CNN2QuantileLightningModule(pl.LightningModule):
    def __init__(
        self,
        num_features: int,
        sequence_length: int,
        output_dim: int,
        quantiles: list[float],
        conv_out_channels: int = 64,
        kernel_size: int = 3,
        hidden_dim1: int = 128,
        hidden_dim2: int = 64,
        dropout: float = 0.1,
        activation: str = "relu",
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        optimizer_name: str = "adamw",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.quantiles = quantiles
        self.model = CNN2QuantileNet(
            num_features=num_features,
            sequence_length=sequence_length,
            output_dim=output_dim,
            quantiles=quantiles,
            conv_out_channels=conv_out_channels,
            kernel_size=kernel_size,
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2,
            dropout=dropout,
            activation=activation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def quantile_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if y.ndim == 1:
            y = y.unsqueeze(-1)
        if y.ndim == 2:
            y = y.unsqueeze(-1)
        qs = y_hat.new_tensor(self.quantiles).view(1, 1, -1)
        err = y - y_hat
        return torch.maximum(qs * err, (qs - 1.0) * err).mean()

    def _p50(self, y_hat: torch.Tensor) -> torch.Tensor:
        if 0.5 not in self.quantiles:
            raise ValueError(f"0.5 quantile is required, got {self.quantiles}")
        return y_hat[:, :, self.quantiles.index(0.5)]

    def _shared_step(self, batch, stage: str):
        x, y, _ = batch
        y_hat = self(x)
        loss = self.quantile_loss(y_hat, y)

        y_p50 = self._p50(y_hat)
        mae = F.l1_loss(y_p50, y)
        rmse = torch.sqrt(F.mse_loss(y_p50, y))

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_mae", mae, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_rmse", rmse, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, meta = batch
        y_hat = self(x)
        return {
            "y_hat": y_hat.detach().cpu(),
            "y": y.detach().cpu(),
            "meta": meta,
        }

    def configure_optimizers(self):
        if self.hparams.optimizer_name.lower() == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }


def _postprocess_predictions(
    pred_outputs,
    df_full: pd.DataFrame,
    scaler: StandardScaler,
    target_idx: int,
    quantiles: list[float],
    time_idx_col: str = "time_idx",
    timestamp_col: str = "timestamp",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if quantiles != [0.1, 0.5, 0.9]:
        raise ValueError("run_all_cnn2.py currently expects quantiles [0.1, 0.5, 0.9].")

    time_map = (
        df_full[[time_idx_col, timestamp_col]]
        .drop_duplicates(subset=[time_idx_col])
        .sort_values(time_idx_col)
        .set_index(time_idx_col)[timestamp_col]
    )

    rows = []
    sample_offset = 0

    for batch_out in pred_outputs:
        y_hat_scaled = batch_out["y_hat"].numpy()
        y_scaled = batch_out["y"].numpy()
        meta = batch_out["meta"]

        dec_start = np.asarray(meta["decoder_time_idx_start"]).astype(np.int64)
        batch_size, horizon, num_quantiles = y_hat_scaled.shape

        if num_quantiles != len(quantiles):
            raise ValueError(
                f"num_quantiles={num_quantiles}, len(quantiles)={len(quantiles)}"
            )

        y_hat = _inverse_target(y_hat_scaled, scaler, target_idx)
        y = _inverse_target(y_scaled, scaler, target_idx)

        for i in range(batch_size):
            sample_id = sample_offset + i
            decoder_time_idx = dec_start[i] + np.arange(horizon, dtype=np.int64)
            ts = pd.to_datetime(time_map.loc[decoder_time_idx].to_numpy())
            forecast_start = ts[0]

            for h in range(horizon):
                rows.append(
                    {
                        "sample": sample_id,
                        "forecast_start": forecast_start,
                        "horizon_idx": h + 1,
                        "timestamp": ts[h],
                        "date": pd.Timestamp(ts[h]).floor("D"),
                        "hour": pd.Timestamp(ts[h]).hour,
                        "true": float(y[i, h]),
                        "pred_q10": float(y_hat[i, h, 0]),
                        "pred_q50": float(y_hat[i, h, 1]),
                        "pred_q90": float(y_hat[i, h, 2]),
                    }
                )

        sample_offset += batch_size

    final_df = pd.DataFrame(rows)
    for col in ["pred_q10", "pred_q50", "pred_q90"]:
        final_df[col] = final_df[col].clip(lower=0)

    final_df["err"] = final_df["pred_q50"] - final_df["true"]
    final_df["abs_err"] = final_df["err"].abs()
    final_df["sq_err"] = final_df["err"] ** 2
    final_df["width"] = final_df["pred_q90"] - final_df["pred_q10"]
    final_df["in_80"] = (
        (final_df["true"] >= final_df["pred_q10"])
        & (final_df["true"] <= final_df["pred_q90"])
    ).astype(int)
    final_df = final_df.sort_values(["date", "timestamp", "sample"]).reset_index(drop=True)

    one = (
        final_df
        .groupby("timestamp", as_index=False)
        .agg(
            true=("true", "first"),
            pred_q10=("pred_q10", "mean"),
            pred_q50=("pred_q50", "mean"),
            pred_q90=("pred_q90", "mean"),
        )
    )
    one["date"] = one["timestamp"].dt.floor("D")
    one["hour"] = one["timestamp"].dt.hour
    one["err"] = one["pred_q50"] - one["true"]
    one["abs_err"] = one["err"].abs()
    one["sq_err"] = one["err"] ** 2
    one["in_80"] = (
        (one["true"] >= one["pred_q10"])
        & (one["true"] <= one["pred_q90"])
    ).astype(int)

    return final_df, one


def _plot_epoch_metrics(run_dir: Path) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        warnings.warn(f"Skipping epoch metric plot because matplotlib is unavailable: {exc}")
        return None

    candidates = sorted(run_dir.glob("lightning_logs/version_*/metrics.csv"))
    if not candidates:
        warnings.warn(f"Skipping epoch metric plot because no metrics.csv was found under {run_dir}")
        return None

    metrics_csv = candidates[-1]
    df = pd.read_csv(metrics_csv)
    if "epoch" not in df.columns:
        warnings.warn(f"Skipping epoch metric plot because 'epoch' column is missing in {metrics_csv}")
        return None

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
        warnings.warn(f"Skipping epoch metric plot because no known metric columns exist in {metrics_csv}")
        return None

    epoch_df = df.groupby("epoch", as_index=False)[metric_cols].last()
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

    out_path = run_dir / "figures" / "epoch_metrics.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def main(
    config_path: str,
    *,
    model_module_cls= CNN2QuantileLightningModule,
    run_suffix: str = "_cnn2_all",
    model_overrides: dict[str, Any] | None = None,
) -> None:
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    seed_all(cfg["run"]["seed"])

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    run_dir = make_run_dir(cfg["run"]["reports_dir"], cfg["run"]["name"] + run_suffix)
    ckpt_dir = run_dir / "checkpoints"
    table_dir = run_dir / "tables"
    meta_dir = run_dir / "meta"

    ensure_dir(run_dir)
    ensure_dir(ckpt_dir)
    ensure_dir(table_dir)
    ensure_dir(meta_dir)
    copy_config(config_path, run_dir)

    df_full = build_df_full(data_cfg)
    split = _resolve_split(cfg, df_full)

    target_col = data_cfg["target_col"]
    group_col = data_cfg.get("group_col", "series_id")
    time_idx_col = data_cfg.get("time_idx_col", "time_idx")
    timestamp_col = "timestamp"

    base_feature_cols = list(data_cfg["feature_cols"])
    model_feature_cols = _unique_keep_order(base_feature_cols + [target_col])
    known_future_cols = data_cfg.get("known_future_cols", base_feature_cols)
    known_future_cols = [col for col in known_future_cols if col != target_col]

    missing = [
        col
        for col in [group_col, time_idx_col, timestamp_col, target_col] + model_feature_cols
        if col not in df_full.columns
    ]
    if missing:
        raise ValueError(f"Missing columns for CNN2 run: {missing}")

    scaler = StandardScaler()
    df_scaled = df_full.copy()
    train_cutoff = split["train_cutoff"]
    scaler.fit(df_full.iloc[:train_cutoff][model_feature_cols])
    df_scaled[model_feature_cols] = scaler.transform(df_full[model_feature_cols])
    target_idx = model_feature_cols.index(target_col)

    dm = CNN2KnownFutureDataModule(
        df_full=df_scaled,
        model_feature_cols=model_feature_cols,
        known_future_cols=known_future_cols,
        target_col=target_col,
        group_col=group_col,
        time_idx_col=time_idx_col,
        encoder_length=data_cfg["encoder_length"],
        prediction_length=data_cfg["prediction_length"],
        training_cutoff=split["training_cutoff"],
        validation_cutoff=split["validation_cutoff"],
        test_start_time_idx=split["test_start_time_idx"],
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg.get("num_workers", 0),
        allow_missing_time_idx=data_cfg.get("allow_missing_time_idx", False),
    )

    dm.setup("fit")
    xb, yb, meta = next(iter(dm.train_dataloader()))
    print("train x shape:", xb.shape)
    print("train y shape:", yb.shape)
    print("example meta:", meta)
    print("model_feature_cols:", model_feature_cols)
    print("known_future_cols:", known_future_cols)

    model_kwargs = {
        "num_features": len(model_feature_cols) + 1,
        "sequence_length": data_cfg["encoder_length"] + data_cfg["prediction_length"],
        "output_dim": data_cfg["prediction_length"],
        "quantiles": train_cfg.get("quantiles", [0.1, 0.5, 0.9]),
        "conv_out_channels": model_cfg.get("conv_out_channels", 64),
        "kernel_size": model_cfg.get("kernel_size", 3),
        "hidden_dim1": _first_present(
            model_cfg, ["cnn2_hidden_dim1", "hidden_dim1"], 128
        ),
        "hidden_dim2": _first_present(
            model_cfg, ["cnn2_hidden_dim2", "hidden_dim2"], 64
        ),
        "activation": model_cfg.get("activation", "relu"),
        "dropout": model_cfg.get("dropout", 0.1),
        "lr": _first_present(
            train_cfg, ["cnn2_lr", "lr", "learning_rate"], 1e-3
        ),
        "weight_decay": train_cfg.get("weight_decay", 1e-5),
        "optimizer_name": _first_present(
            train_cfg, ["cnn2_optimizer_name", "optimizer_name"], "adamw"
        ),
    }
    if model_overrides:
        model_kwargs.update(model_overrides)

    model = model_module_cls(
        **model_kwargs,
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=train_cfg.get("early_stop_patience", 10),
            mode="min",
        ),
        ckpt_cb,
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs=train_cfg.get("max_epochs", 50),
        accelerator=train_cfg.get("accelerator", "auto"),
        devices=train_cfg.get("devices", 1),
        callbacks=callbacks,
        log_every_n_steps=train_cfg.get("log_every_n_steps", 10),
        default_root_dir=str(run_dir),
        deterministic=train_cfg.get("deterministic", "warn"),
        gradient_clip_val=train_cfg.get("gradient_clip_val"),
    )

    trainer.fit(model, datamodule=dm)
    epoch_plot_path = _plot_epoch_metrics(run_dir)

    best_ckpt = ckpt_cb.best_model_path or ckpt_cb.last_model_path
    if not best_ckpt:
        best_ckpt = str(ckpt_dir / "last.ckpt")
        trainer.save_checkpoint(best_ckpt)

    dm.setup("test")
    train_meta = {
        "run_dir": str(run_dir),
        "best_ckpt": best_ckpt,
        "n_total": split["n_total"],
        "train_cutoff": split["train_cutoff"],
        "val_cutoff": split["val_cutoff"],
        "train_cutoff_time_idx": split["train_cutoff_time_idx"],
        "val_cutoff_time_idx": split["val_cutoff_time_idx"],
        "test_start_time_idx": split["test_start_time_idx"],
        "test_end_time_idx": split["test_end_time_idx"],
        "train_ratio": split["train_ratio"],
        "val_ratio": split["val_ratio"],
        "test_ratio": split["test_ratio"],
        "n_train_rows": split["train_rows"],
        "n_val_rows": split["val_rows"],
        "n_test_rows": split["test_rows"],
        "n_train_samples": len(dm.train_dataset),
        "n_val_samples": len(dm.val_dataset),
        "n_test_samples": len(dm.test_dataset),
        "model_feature_cols": model_feature_cols,
        "known_future_cols": known_future_cols,
        "target_scaler_mean": float(scaler.mean_[target_idx]),
        "target_scaler_scale": float(scaler.scale_[target_idx]),
        "effective_model_hparams": {
            "conv_out_channels": int(model.hparams.conv_out_channels),
            "kernel_size": int(model.hparams.kernel_size),
            "hidden_dim1": int(model.hparams.hidden_dim1),
            "hidden_dim2": int(model.hparams.hidden_dim2),
            "dropout": float(model.hparams.dropout),
            "lr": float(model.hparams.lr),
            "optimizer_name": str(model.hparams.optimizer_name),
            "weight_decay": float(model.hparams.weight_decay),
        },
    }
    save_json(train_meta, meta_dir / "train_meta.json")

    best_model = model_module_cls.load_from_checkpoint(best_ckpt)
    pred_trainer = pl.Trainer(
        accelerator=train_cfg.get("accelerator", "auto"),
        devices=train_cfg.get("devices", 1),
        default_root_dir=str(run_dir),
        logger=False,
        enable_checkpointing=False,
    )

    pred_outputs = pred_trainer.predict(best_model, dataloaders=dm.test_dataloader())
    final_df, one = _postprocess_predictions(
        pred_outputs,
        df_full=df_full,
        scaler=scaler,
        target_idx=target_idx,
        quantiles=train_cfg.get("quantiles", [0.1, 0.5, 0.9]),
        time_idx_col=time_idx_col,
        timestamp_col=timestamp_col,
    )

    expected_horizon = int(data_cfg["prediction_length"])
    actual_horizon = int(final_df["horizon_idx"].max())
    if actual_horizon != expected_horizon:
        raise ValueError(
            f"Config prediction_length={expected_horizon} does not match "
            f"postprocessed horizon={actual_horizon}."
        )

    horizon_metrics = (
        final_df
        .groupby("horizon_idx", as_index=False)
        .agg(
            mae=("abs_err", "mean"),
            mse=("sq_err", "mean"),
            coverage_80=("in_80", "mean"),
            mean_width=("width", "mean"),
        )
    )
    horizon_metrics["rmse"] = horizon_metrics["mse"] ** 0.5
    horizon_metrics = horizon_metrics[["horizon_idx", "mae", "rmse", "coverage_80", "mean_width"]]

    out_one = table_dir / "pred_one.csv"
    out_all = table_dir / "pred_all_samples.csv"
    out_horizon_metrics = table_dir / "horizon_metrics.csv"
    one.to_csv(out_one, index=False, encoding="utf-8-sig")
    final_df.to_csv(out_all, index=False, encoding="utf-8-sig")
    horizon_metrics.to_csv(out_horizon_metrics, index=False, encoding="utf-8-sig")

    save_json(
        {
            "prediction_length": expected_horizon,
            "observed_horizon": actual_horizon,
            "horizon_metrics": {
                str(int(row["horizon_idx"])): {
                    "mae": float(row["mae"]),
                    "rmse": float(row["rmse"]),
                    "coverage_80": float(row["coverage_80"]),
                    "mean_width": float(row["mean_width"]),
                }
                for _, row in horizon_metrics.iterrows()
            },
        },
        meta_dir / "predict_meta.json",
    )

    eval_cnn_main(config_path, str(run_dir))

    print("RUN_DIR:", run_dir)
    print("BEST_CKPT:", best_ckpt)
    print("SAVED:", out_one)
    print("SAVED:", out_all)
    print("SAVED:", out_horizon_metrics)
    print("SAVED:", run_dir / "reports" / "metrics.json")
    if epoch_plot_path is not None:
        print("SAVED:", epoch_plot_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
