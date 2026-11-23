# scripts/run_all_lstm.py
from __future__ import annotations

import argparse
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
from pv_forecast.metrics.coverage import global_metrics, hourly_coverage


def _unique_keep_order(cols: list[str]) -> list[str]:
    return list(dict.fromkeys(cols))


def _inverse_target(values: np.ndarray, scaler: StandardScaler, target_idx: int) -> np.ndarray:
    return values * scaler.scale_[target_idx] + scaler.mean_[target_idx]


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
class LSTMRunSample:
    x: np.ndarray
    y: np.ndarray
    group_id: str
    decoder_time_idx_start: int
    decoder_time_idx_end: int


class LSTMKnownFutureDataset(Dataset):
    """
    LSTM baseline aligned more closely with TFT's information setup.

    Encoder rows contain all model features, including past target.
    Decoder rows contain only known future features; unknown future channels,
    including target, are zero in scaled space. A decoder indicator channel is
    appended so the LSTM can distinguish historical and future-known rows.
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

        self.samples: list[LSTMRunSample] = []
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
                    LSTMRunSample(
                        x=x,
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


class LSTMKnownFutureDataModule(pl.LightningDataModule):
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
            self.train_dataset = LSTMKnownFutureDataset(split="train", **self.kwargs)
            self.val_dataset = LSTMKnownFutureDataset(split="val", **self.kwargs)
        if stage in ("test", "predict"):
            self.test_dataset = LSTMKnownFutureDataset(split="test", **self.kwargs)

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


class LSTMQuantileForecaster(pl.LightningModule):
    """Causal multi-horizon LSTM quantile forecaster.

    The input contains encoder history followed by decoder rows with only
    future-known covariates. Each horizon is predicted from its corresponding
    decoder hidden state, so horizon ``h`` cannot use covariates after ``h``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        output_dim: int,
        quantiles: list[float],
        lr: float,
        weight_decay: float = 0.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.quantiles = quantiles
        self.output_dim = output_dim
        self.num_quantiles = len(quantiles)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # ``nn.LSTM.dropout`` is active only between stacked recurrent layers.
        # Apply the configured rate to decoder representations as well so the
        # hyperparameter remains effective when ``num_layers == 1``.
        self.output_dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, self.num_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        if out.size(1) < self.output_dim:
            raise ValueError(
                f"sequence length ({out.size(1)}) must be at least "
                f"output_dim ({self.output_dim})"
            )

        # The final output_dim positions are decoder positions. A unidirectional
        # LSTM state at horizon h has seen encoder history and future-known
        # covariates only through h, matching causal TFT/Mamba-TFT semantics.
        decoder_out = out[:, -self.output_dim :, :]
        decoder_out = self.output_dropout(decoder_out)
        return self.fc(decoder_out)

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
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )


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
        raise ValueError("run_all_lstm.py currently expects quantiles [0.1, 0.5, 0.9].")

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


def main(config_path: str) -> None:
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    seed_all(cfg["run"]["seed"])

    data_cfg = cfg["data"]
    lstm_cfg = cfg["lstm"]

    run_dir = make_run_dir(cfg["run"]["reports_dir"], cfg["run"]["name"] + "_lstm_all")
    ckpt_dir = run_dir / "checkpoints"
    table_dir = run_dir / "tables"
    meta_dir = run_dir / "meta"

    ensure_dir(run_dir)
    ensure_dir(ckpt_dir)
    ensure_dir(table_dir)
    ensure_dir(meta_dir)
    copy_config(config_path, run_dir)

    df_full = build_df_full(data_cfg)
    if data_cfg.get("index_is_time", False):
        df_full = df_full.reset_index()
        if "timestamp" not in df_full.columns:
            df_full = df_full.rename(columns={df_full.columns[0]: "timestamp"})

    split = _resolve_split(cfg, df_full)

    target_col = cfg["features"]["target"]
    group_col = data_cfg.get("group_col", "series_id")
    time_idx_col = data_cfg.get("time_idx_col", "time_idx")
    timestamp_col = "timestamp"

    base_feature_cols = list(cfg["features"]["input_reals"])
    model_feature_cols = _unique_keep_order(base_feature_cols + [target_col])
    known_future_cols = cfg["features"].get("known_future_reals", base_feature_cols)
    known_future_cols = [col for col in known_future_cols if col != target_col]

    missing = [
        col
        for col in [group_col, time_idx_col, timestamp_col, target_col] + model_feature_cols
        if col not in df_full.columns
    ]
    if missing:
        raise ValueError(f"Missing columns for LSTM run: {missing}")

    scaler = StandardScaler()
    df_scaled = df_full.copy()
    train_cutoff = split["train_cutoff"]
    scaler.fit(df_full.iloc[:train_cutoff][model_feature_cols])
    df_scaled[model_feature_cols] = scaler.transform(df_full[model_feature_cols])
    target_idx = model_feature_cols.index(target_col)

    encoder_length = int(lstm_cfg["window_size"])
    prediction_length = int(lstm_cfg["horizon"])

    dm = LSTMKnownFutureDataModule(
        df_full=df_scaled,
        model_feature_cols=model_feature_cols,
        known_future_cols=known_future_cols,
        target_col=target_col,
        group_col=group_col,
        time_idx_col=time_idx_col,
        encoder_length=encoder_length,
        prediction_length=prediction_length,
        training_cutoff=split["training_cutoff"],
        validation_cutoff=split["validation_cutoff"],
        test_start_time_idx=split["test_start_time_idx"],
        batch_size=lstm_cfg["batch_size"],
        num_workers=lstm_cfg.get("num_workers", 0),
        allow_missing_time_idx=data_cfg.get("allow_missing_time_idx", False),
    )

    dm.setup("fit")
    xb, yb, meta = next(iter(dm.train_dataloader()))
    print("train x shape:", xb.shape)
    print("train y shape:", yb.shape)
    print("example meta:", meta)
    print("model_feature_cols:", model_feature_cols)
    print("known_future_cols:", known_future_cols)

    quantiles = lstm_cfg.get("quantiles", [0.1, 0.5, 0.9])
    model = LSTMQuantileForecaster(
        input_size=len(model_feature_cols) + 1,
        hidden_size=lstm_cfg["hidden_size"],
        num_layers=lstm_cfg["num_layers"],
        dropout=lstm_cfg["dropout"],
        output_dim=prediction_length,
        quantiles=quantiles,
        lr=lstm_cfg.get("run_all_lr", lstm_cfg.get("learning_rate", 1e-3)),
        weight_decay=lstm_cfg.get("weight_decay", 1e-5),
    )

    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best-{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )

    trainer = pl.Trainer(
        max_epochs=lstm_cfg.get("max_epochs", 50),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[
            EarlyStopping(
                monitor="val_loss",
                patience=lstm_cfg.get("early_stop_patience", 5),
                mode="min",
            ),
            LearningRateMonitor(logging_interval="epoch"),
            ckpt_cb,
        ],
        default_root_dir=str(run_dir),
        deterministic=lstm_cfg.get("deterministic", "warn"),
        gradient_clip_val=lstm_cfg.get("gradient_clip_val"),
    )

    trainer.fit(model, datamodule=dm)

    best_ckpt = ckpt_cb.best_model_path or ckpt_cb.last_model_path
    if not best_ckpt:
        best_ckpt = str(ckpt_dir / "last.ckpt")
        trainer.save_checkpoint(best_ckpt)

    dm.setup("test")
    train_meta = {
        "run_dir": str(run_dir),
        "best_ckpt": best_ckpt,
        "model_architecture": "causal_decoder_hidden_per_horizon",
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
    }
    save_json(train_meta, meta_dir / "train_meta.json")

    best_model = LSTMQuantileForecaster.load_from_checkpoint(best_ckpt)
    pred_trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
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
        quantiles=quantiles,
        time_idx_col=time_idx_col,
        timestamp_col=timestamp_col,
    )

    expected_horizon = prediction_length
    actual_horizon = int(final_df["horizon_idx"].max())
    if actual_horizon != expected_horizon:
        raise ValueError(
            f"Config horizon={expected_horizon} does not match "
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

    out_all = table_dir / "pred_results.csv"
    out_one = table_dir / "pred_results_by_timestamp.csv"
    out_horizon_metrics = table_dir / "horizon_metrics.csv"
    final_df.to_csv(out_all, index=False, encoding="utf-8-sig")
    one.to_csv(out_one, index=False, encoding="utf-8-sig")
    horizon_metrics.to_csv(out_horizon_metrics, index=False, encoding="utf-8-sig")

    metrics = global_metrics(one)
    metrics["split"] = {
        "train_ratio": split["train_ratio"],
        "val_ratio": split["val_ratio"],
        "test_ratio": split["test_ratio"],
        "train_cutoff": split["train_cutoff"],
        "val_cutoff": split["val_cutoff"],
        "train_cutoff_time_idx": split["train_cutoff_time_idx"],
        "val_cutoff_time_idx": split["val_cutoff_time_idx"],
    }
    metrics["horizon_metrics"] = {
        str(int(row["horizon_idx"])): {
            "mae": float(row["mae"]),
            "rmse": float(row["rmse"]),
            "coverage_80": float(row["coverage_80"]),
            "mean_width": float(row["mean_width"]),
        }
        for _, row in horizon_metrics.iterrows()
    }
    save_json(metrics, meta_dir / "metrics.json")
    save_json(
        {
            "ckpt": best_ckpt,
            "prediction_length": expected_horizon,
            "observed_horizon": actual_horizon,
            "horizon_metrics": metrics["horizon_metrics"],
        },
        meta_dir / "predict_meta.json",
    )

    hourly = hourly_coverage(one, cfg["eval"]["hours_from"], cfg["eval"]["hours_to"])
    hourly_df = hourly.rename("coverage_pct").reset_index().rename(columns={"index": "hour"})
    hourly_df.to_csv(table_dir / "hourly_coverage.csv", index=False, encoding="utf-8-sig")

    print("RUN_DIR:", run_dir)
    print("BEST_CKPT:", best_ckpt)
    print("SAVED:", out_one)
    print("SAVED:", out_all)
    print("SAVED:", out_horizon_metrics)
    print("SAVED:", meta_dir / "metrics.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
