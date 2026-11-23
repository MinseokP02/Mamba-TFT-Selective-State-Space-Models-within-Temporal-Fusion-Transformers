# scripts/run_all2.py
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import yaml
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

from pv_forecast.common.paths import copy_config, make_run_dir, save_json
from pv_forecast.common.seed import seed_all
from pv_forecast.data.build_df_full import build_df_full
from pv_forecast.metrics.coverage import global_metrics, hourly_coverage
from pv_forecast.post.extract import extract_pred_tensor


DETERMINISTIC_KNOWN_REALS = {
    "time_idx",
    "hour",
    "dayofweek",
    "month",
    "dayofyear",
    "year",
    "is_daylight",
}


def _unique_keep_order(cols: list[str]) -> list[str]:
    return list(dict.fromkeys(cols))


def _resolve_known_unknown_reals(cfg: dict[str, Any], df_full: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    feats = cfg["features"]

    requested_known = list(feats.get("known_reals", []))
    requested_unknown = list(feats.get("unknown_reals", []))
    future_available = set(feats.get("future_available_reals", []))

    allowed_known = DETERMINISTIC_KNOWN_REALS | future_available

    safe_known = [
        col
        for col in requested_known
        if col in allowed_known and col in df_full.columns
    ]
    excluded_known = [
        col
        for col in requested_known
        if col not in safe_known and col in df_full.columns
    ]

    target = feats["target"]
    unknown = _unique_keep_order(requested_unknown + excluded_known)
    if target not in unknown:
        unknown.append(target)
    unknown = [col for col in unknown if col in df_full.columns and col not in safe_known]

    missing_known = [col for col in safe_known if col not in df_full.columns]
    missing_unknown = [col for col in unknown if col not in df_full.columns]
    if missing_known or missing_unknown:
        raise ValueError(
            f"Missing TFT covariate columns. known={missing_known}, unknown={missing_unknown}"
        )

    if excluded_known:
        warnings.warn(
            "Moved these requested known_reals to encoder-only unknown_reals because "
            "they are not listed in features.future_available_reals: "
            f"{excluded_known}",
            stacklevel=2,
        )

    return safe_known, unknown, excluded_known


def _build_training_dataset(
    df_train: pd.DataFrame,
    cfg: dict[str, Any],
    known_reals: list[str],
    unknown_reals: list[str],
) -> TimeSeriesDataSet:
    feats = cfg["features"]
    tft = cfg["tft"]

    return TimeSeriesDataSet(
        df_train,
        time_idx="time_idx",
        target=feats["target"],
        group_ids=feats["group_ids"],
        max_encoder_length=tft["max_encoder_length"],
        max_prediction_length=tft["max_prediction_length"],
        time_varying_known_reals=known_reals,
        time_varying_unknown_reals=unknown_reals,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )


def _build_tft(training_ds: TimeSeriesDataSet, cfg: dict[str, Any]) -> TemporalFusionTransformer:
    m = cfg["tft"]
    quantiles = m.get("quantiles", [0.1, 0.5, 0.9])

    return TemporalFusionTransformer.from_dataset(
        training_ds,
        hidden_size=m["hidden_size"],
        lstm_layers=m["lstm_layers"],
        dropout=m["dropout"],
        attention_head_size=m["attention_head_size"],
        learning_rate=m["learning_rate"],
        loss=QuantileLoss(quantiles),
    )


def _quantile_index(quantiles: list[float], q: float) -> int:
    for idx, value in enumerate(quantiles):
        if np.isclose(float(value), q):
            return idx
    raise ValueError(f"Required quantile {q} not found in quantiles={quantiles}")


def _postprocess_raw_to_frames(
    raw,
    df_full: pd.DataFrame,
    quantiles: list[float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output, x, index, decoder_lengths, y = raw
    pred_tensor = extract_pred_tensor(output)

    q10_idx = _quantile_index(quantiles, 0.1)
    q50_idx = _quantile_index(quantiles, 0.5)
    q90_idx = _quantile_index(quantiles, 0.9)

    dec_ti = x["decoder_time_idx"].detach().cpu().numpy().astype(np.int64)
    true = x["decoder_target"].detach().cpu().numpy()
    pred = pred_tensor.detach().cpu().numpy()

    if pred.ndim != 3:
        raise ValueError(f"Expected prediction tensor shape (N,H,Q), got {pred.shape}")

    time_map = (
        df_full[["time_idx", "timestamp"]]
        .drop_duplicates(subset=["time_idx"])
        .sort_values("time_idx")
        .set_index("time_idx")["timestamp"]
    )

    n_samples, horizon = dec_ti.shape
    ts_flat = pd.to_datetime(time_map.loc[dec_ti.reshape(-1)].to_numpy())

    out_all = pd.DataFrame(
        {
            "sample": np.repeat(np.arange(n_samples), horizon),
            "horizon": np.tile(np.arange(1, horizon + 1), n_samples),
            "horizon_idx": np.tile(np.arange(1, horizon + 1), n_samples),
            "timestamp": ts_flat,
            "date": ts_flat.floor("D"),
            "hour": ts_flat.hour,
            "true": true.reshape(-1),
            "pred_q10": pred[:, :, q10_idx].reshape(-1),
            "pred_q50": pred[:, :, q50_idx].reshape(-1),
            "pred_q90": pred[:, :, q90_idx].reshape(-1),
        }
    )

    for col in ["pred_q10", "pred_q50", "pred_q90"]:
        out_all[col] = out_all[col].clip(lower=0)

    out_all["err"] = out_all["pred_q50"] - out_all["true"]
    out_all["abs_err"] = out_all["err"].abs()
    out_all["sq_err"] = out_all["err"] ** 2
    out_all["width"] = out_all["pred_q90"] - out_all["pred_q10"]
    out_all["in_80"] = (
        (out_all["true"] >= out_all["pred_q10"])
        & (out_all["true"] <= out_all["pred_q90"])
    ).astype(int)

    final_df = out_all.sort_values(["date", "timestamp", "sample"]).reset_index(drop=True)

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
    one["hour"] = one["timestamp"].dt.hour
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
    dataframe_transform=None,
    run_suffix: str = "_all2",
) -> None:
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    seed_all(cfg["run"]["seed"])

    run_dir = make_run_dir(cfg["run"]["reports_dir"], cfg["run"]["name"] + run_suffix)
    copy_config(config_path, run_dir)

    df_full = build_df_full(cfg["data"])
    transform_meta: dict[str, Any] = {}
    if dataframe_transform is not None:
        result = dataframe_transform(df_full, cfg)
        if result is not None:
            transform_meta = dict(result)
    n = len(df_full)

    train_ratio = float(cfg["split"]["train_ratio"])
    val_ratio = float(cfg["split"]["val_ratio"])
    test_ratio = 1.0 - train_ratio - val_ratio

    if not (0 < train_ratio < 1):
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    if not (0 <= val_ratio < 1):
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    if test_ratio <= 0:
        raise ValueError(
            f"Invalid split ratios: train_ratio({train_ratio}) + "
            f"val_ratio({val_ratio}) must be < 1.0 so test split remains."
        )

    train_cutoff = int(n * train_ratio)
    val_cutoff = int(n * (train_ratio + val_ratio))

    df_train = df_full.iloc[:train_cutoff].copy()
    train_cutoff_time_idx = int(df_full.iloc[train_cutoff]["time_idx"])
    val_cutoff_time_idx = int(df_full.iloc[val_cutoff]["time_idx"])

    df_val = df_full[df_full["time_idx"] < val_cutoff_time_idx].copy()
    df_test = df_full.copy()

    known_reals, unknown_reals, excluded_known_reals = _resolve_known_unknown_reals(cfg, df_full)

    print("known_reals used by decoder:", known_reals)
    print("unknown_reals used by encoder:", unknown_reals)
    if excluded_known_reals:
        print("excluded from decoder known_reals:", excluded_known_reals)

    training_ds = _build_training_dataset(df_train, cfg, known_reals, unknown_reals)
    val_ds = TimeSeriesDataSet.from_dataset(
        training_ds,
        df_val,
        min_prediction_idx=train_cutoff_time_idx,
        stop_randomization=True,
    )
    test_ds = TimeSeriesDataSet.from_dataset(
        training_ds,
        df_test,
        min_prediction_idx=val_cutoff_time_idx,
        stop_randomization=True,
    )

    if len(val_ds) == 0:
        raise ValueError(
            "Validation dataset is empty. "
            "Check split ratios, max_encoder_length, max_prediction_length, and time_idx continuity."
        )
    if len(test_ds) == 0:
        raise ValueError(
            "Test dataset is empty. "
            "Check split ratios, max_encoder_length, max_prediction_length, and time_idx continuity."
        )

    train_loader = training_ds.to_dataloader(
        train=True,
        batch_size=cfg["tft"]["batch_size"],
        num_workers=0,
    )
    val_loader = val_ds.to_dataloader(
        train=False,
        batch_size=cfg["tft"]["batch_size"],
        num_workers=0,
    )
    test_loader = test_ds.to_dataloader(
        train=False,
        batch_size=cfg["tft"]["batch_size"],
        num_workers=0,
    )

    model = _build_tft(training_ds, cfg)

    ckpt_dir = Path(cfg["run"]["artifacts_dir"]) / cfg["run"]["name"]
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        save_top_k=1,
        monitor="val_loss",
        mode="min",
        filename="{epoch}-{val_loss:.4f}",
    )

    trainer = pl.Trainer(
        max_epochs=cfg["tft"]["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=cfg["tft"].get("early_stop_patience", 5), mode="min"),
            LearningRateMonitor(logging_interval="epoch"),
            ckpt_cb,
        ],
        default_root_dir=str(run_dir),
        num_sanity_val_steps=2,
        deterministic="warn",
        gradient_clip_val=cfg["tft"].get("gradient_clip_val", 0.1),
    )

    trainer.fit(model, train_loader, val_loader)
    epoch_plot_path = _plot_epoch_metrics(run_dir)

    best_ckpt = ckpt_cb.best_model_path or ckpt_cb.last_model_path
    if not best_ckpt:
        best_ckpt = str(ckpt_dir / "last.ckpt")
        trainer.save_checkpoint(best_ckpt)

    save_json(
        {
            "run_dir": str(run_dir),
            "best_ckpt": best_ckpt,
            "n_total": n,
            "train_cutoff": train_cutoff,
            "val_cutoff": val_cutoff,
            "train_cutoff_time_idx": train_cutoff_time_idx,
            "val_cutoff_time_idx": val_cutoff_time_idx,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "n_train_rows": len(df_train),
            "n_val_samples": len(val_ds),
            "n_test_samples": len(test_ds),
            "known_reals": known_reals,
            "unknown_reals": unknown_reals,
            "excluded_known_reals": excluded_known_reals,
            "dataframe_transform": transform_meta,
        },
        run_dir / "meta" / "train_meta.json",
    )

    best_val_loss = (
        float(ckpt_cb.best_model_score.detach().cpu())
        if ckpt_cb.best_model_score is not None
        else None
    )
    save_json(
        {
            "selection_metric": "val_loss",
            "selection_split": "validation",
            "val_loss": best_val_loss,
            "best_ckpt": best_ckpt,
        },
        run_dir / "meta" / "selection_metrics.json",
    )
    if not cfg.get("run", {}).get("evaluate_test", True):
        print("RUN_DIR:", run_dir)
        print("BEST_CKPT:", best_ckpt)
        print("BEST_VAL_LOSS:", best_val_loss)
        print("TEST EVALUATION: skipped")
        return

    best_model = TemporalFusionTransformer.load_from_checkpoint(best_ckpt)
    raw = best_model.predict(test_loader, mode="raw", return_x=True)
    final_df, one = _postprocess_raw_to_frames(
        raw,
        df_full,
        quantiles=cfg["tft"].get("quantiles", [0.1, 0.5, 0.9]),
    )

    expected_horizon = int(cfg["tft"]["max_prediction_length"])
    actual_horizon = int(final_df["horizon_idx"].max())
    if actual_horizon != expected_horizon:
        raise ValueError(
            f"Config max_prediction_length={expected_horizon} does not match "
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

    out_one = run_dir / "tables" / "pred_one.csv"
    out_all = run_dir / "tables" / "pred_all_samples.csv"
    out_horizon_metrics = run_dir / "tables" / "horizon_metrics.csv"
    one.to_csv(out_one, index=False, encoding="utf-8-sig")
    final_df.to_csv(out_all, index=False, encoding="utf-8-sig")
    horizon_metrics.to_csv(out_horizon_metrics, index=False, encoding="utf-8-sig")

    save_json(
        {
            "max_prediction_length": expected_horizon,
            "observed_horizon": actual_horizon,
            "known_reals": known_reals,
            "unknown_reals": unknown_reals,
            "excluded_known_reals": excluded_known_reals,
            "dataframe_transform": transform_meta,
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
        run_dir / "meta" / "predict_meta.json",
    )

    hourly = hourly_coverage(one, cfg["eval"]["hours_from"], cfg["eval"]["hours_to"])
    hourly_df = hourly.rename("coverage_pct").reset_index().rename(columns={"index": "hour"})
    hourly_df.to_csv(run_dir / "tables" / "hourly_coverage.csv", index=False, encoding="utf-8-sig")

    metrics = global_metrics(one)
    metrics["split"] = {
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "train_cutoff": train_cutoff,
        "val_cutoff": val_cutoff,
        "train_cutoff_time_idx": train_cutoff_time_idx,
        "val_cutoff_time_idx": val_cutoff_time_idx,
    }
    metrics["known_reals"] = known_reals
    metrics["unknown_reals"] = unknown_reals
    metrics["excluded_known_reals"] = excluded_known_reals
    metrics["dataframe_transform"] = transform_meta
    save_json(metrics, run_dir / "meta" / "metrics.json")

    print(f"TOTAL ROWS: {n}")
    print(f"TRAIN ROWS: {len(df_train)}")
    print(f"VAL SAMPLES: {len(val_ds)}")
    print(f"TEST SAMPLES: {len(test_ds)}")
    print(f"TRAIN CUTOFF ROW: {train_cutoff} | time_idx={train_cutoff_time_idx}")
    print(f"VAL   CUTOFF ROW: {val_cutoff} | time_idx={val_cutoff_time_idx}")

    for h, v in hourly.items():
        if v == v:
            print(f"{h:02d}:00 | {v:5.1f}%")

    print("RUN_DIR:", run_dir)
    print("BEST_CKPT:", best_ckpt)
    print("SAVED:", out_one)
    print("SAVED:", out_all)
    print("SAVED:", out_horizon_metrics)
    print("SAVED:", run_dir / "meta" / "metrics.json")
    if epoch_plot_path is not None:
        print("SAVED:", epoch_plot_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
