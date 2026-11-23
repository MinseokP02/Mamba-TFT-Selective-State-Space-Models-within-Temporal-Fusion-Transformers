# scripts/run_all.py
import argparse
from pathlib import Path
import yaml
import torch
import warnings
import pandas as pd
from typing import Any

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet

from pv_forecast.models.tft_vendor import MambaTemporalFusionTransformer

from pv_forecast.common.seed import seed_all
from pv_forecast.common.paths import make_run_dir, copy_config, save_json
from pv_forecast.data.build_df_full import build_df_full
from pv_forecast.data.dataset import build_training_dataset, build_validation_dataset, build_eval_dataset
from pv_forecast.models.mamba_tft import build_mamba_tft
from pv_forecast.post.extract import postprocess_raw_to_frames
from pv_forecast.metrics.coverage import hourly_coverage, global_metrics


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
            "train_loss_step",
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
    loss_cols = [
        col
        for col in ["train_loss_epoch", "train_loss_step", "train_loss", "val_loss"]
        if col in epoch_df.columns
    ]
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
    run_suffix: str = "_all",
):
    # TimeSeriesDataSet fits one StandardScaler per named real column and then
    # transforms a scalar fill value as an ndarray. There is only one feature,
    # so the sklearn feature-name warning does not indicate an order mismatch.
    warnings.filterwarnings(
        "ignore",
        message=(
            r"X does not have valid feature names, but "
            r"StandardScaler was fitted with feature names"
        ),
        category=UserWarning,
    )

    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    seed_all(cfg["run"]["seed"])

    # run_dir 하나로 전 과정을 자동 적재
    run_dir = make_run_dir(cfg["run"]["reports_dir"], cfg["run"]["name"] + run_suffix)
    copy_config(config_path, run_dir)

    # -----------------------
    # 1) DATA
    # -----------------------
    df_full = build_df_full(cfg["data"])
    transform_meta: dict[str, Any] = {}
    if dataframe_transform is not None:
        result = dataframe_transform(df_full, cfg)
        if result is not None:
            transform_meta = dict(result)
    n = len(df_full)

    train_ratio = cfg["split"]["train_ratio"]
    val_ratio = cfg["split"]["val_ratio"]   # config에 추가 필요
    test_ratio = 1.0 - train_ratio - val_ratio


    if not (0 < train_ratio < 1):
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    if not (0 <= val_ratio < 1):
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    if test_ratio <= 0:
        raise ValueError(
            f"Invalid split ratios: train_ratio({train_ratio}) + val_ratio({val_ratio}) "
            f"must be < 1.0 so test split remains."
        )

    train_cutoff = int(n * train_ratio)
    val_cutoff = int(n * (train_ratio + val_ratio))

    df_train = df_full.iloc[:train_cutoff].copy()

    train_cutoff_time_idx = int(df_full.iloc[train_cutoff]["time_idx"])
    val_cutoff_time_idx = int(df_full.iloc[val_cutoff]["time_idx"])

    df_val = df_full[df_full["time_idx"] < val_cutoff_time_idx].copy()
    df_test = df_full.copy()

    # -----------------------
    # 2) DATASET / DATALOADER
    # -----------------------
    training_ds = build_training_dataset(df_train, cfg)
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

    # -----------------------
    # 3) TRAIN + SAVE BEST CKPT
    # -----------------------
    model = build_mamba_tft(training_ds, cfg)

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
        gradient_clip_val=cfg['tft'].get('gradient_clip_val', 0.1)
    )

    trainer.fit(model, train_loader, val_loader)
    epoch_plot_path = _plot_epoch_metrics(run_dir)

    best_ckpt = ckpt_cb.best_model_path or ckpt_cb.last_model_path
    if not best_ckpt:
        # 마지막 수단: 강제 저장 (val_loss 모니터링 실패 같은 상황 대비)
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

    # -----------------------
    # 4) LOAD BEST CKPT -> PREDICT
    # -----------------------
    # Reload with the same concrete architecture that was used for training.
    # SeparateMambaTemporalFusionTransformer checkpoints contain encoder- and
    # decoder-specific constructor arguments that the legacy joint class does
    # not accept.
    best_model = type(model).load_from_checkpoint(best_ckpt)
    raw = best_model.predict(test_loader, mode="raw", return_x=True)

    final_df, one = postprocess_raw_to_frames(raw, df_full)

    if "horizon_idx" not in final_df.columns:
        raise ValueError("Mamba-TFT prediction output is missing horizon_idx; multi-horizon postprocess failed.")

    expected_horizon = int(cfg["tft"]["max_prediction_length"])
    actual_horizon = int(final_df["horizon_idx"].max())
    if actual_horizon != expected_horizon:
        raise ValueError(
            f"Config max_prediction_length={expected_horizon} does not match "
            f"postprocessed horizon={actual_horizon}."
        )

    final_df["width"] = final_df["pred_q90"] - final_df["pred_q10"]
    final_df["in_80"] = (
        (final_df["true"] >= final_df["pred_q10"])
        & (final_df["true"] <= final_df["pred_q90"])
    ).astype(int)

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

    # -----------------------
    # 5) EVAL + SAVE (tables/meta/figures)
    # -----------------------
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
    metrics["dataframe_transform"] = transform_meta
    save_json(metrics, run_dir / "meta" / "metrics.json")

    # 콘솔 출력
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
