# scripts/run_all_cnn.py
from __future__ import annotations
import yaml


import argparse
from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from pv_forecast.common.io import load_yaml, ensure_dir, postprocess_cnn_predictions
from pv_forecast.common.paths import make_run_dir, copy_config, save_json
from pv_forecast.common.seed import seed_all
from pv_forecast.data.build_df_full import build_df_full
from pv_forecast.data.seq_dataset import SequenceDataModule
from pv_forecast.models.cnn1d_quantile import CNN1DQuantileLightningModule
from scripts.eval_cnn_impl import eval_cnn_main


def _resolve_split(cfg: dict, df_full):
    """
    TFT run_all.py와 동일한 방식으로 split 계산.
    - df_full row 기준 train_cutoff / val_cutoff 계산
    - 해당 row의 time_idx를 cut point로 사용
    - test 시작은 val_cutoff_time_idx
    """
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

    test_start_time_idx = val_cutoff_time_idx
    test_end_time_idx = int(df_full[time_idx_col].max())

    train_rows = train_cutoff
    val_rows = val_cutoff - train_cutoff
    test_rows = n - val_cutoff

    return {
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,

        "n_total": n,
        "train_cutoff": train_cutoff,
        "val_cutoff": val_cutoff,

        "train_cutoff_time_idx": train_cutoff_time_idx,
        "val_cutoff_time_idx": val_cutoff_time_idx,

        # SequenceDataModule에 넘길 값
        "training_cutoff": train_cutoff_time_idx,
        "validation_cutoff": val_cutoff_time_idx,
        "test_start_time_idx": test_start_time_idx,
        "test_end_time_idx": test_end_time_idx,

        "train_rows": train_rows,
        "val_rows": val_rows,
        "test_rows": test_rows,
    }


def _build_datamodule(cfg: dict, df_full, split: dict) -> SequenceDataModule:
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    return SequenceDataModule(
        df_full=df_full,
        feature_cols=data_cfg["feature_cols"],
        target_col=data_cfg["target_col"],
        group_col=data_cfg.get("group_col", "series_id"),
        time_idx_col=data_cfg.get("time_idx_col", "time_idx"),
        encoder_length=data_cfg["encoder_length"],
        prediction_length=data_cfg["prediction_length"],

        training_cutoff=split["training_cutoff"],
        validation_cutoff=split["validation_cutoff"],
        test_start_time_idx=split["test_start_time_idx"],

        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg.get("num_workers", 0),
        allow_missing_time_idx=data_cfg.get("allow_missing_time_idx", False),
        model_type="cnn1d",
    )


def _build_model(cfg: dict) -> CNN1DQuantileLightningModule:
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    return CNN1DQuantileLightningModule(
        num_features=len(data_cfg["feature_cols"]),
        encoder_length=data_cfg["encoder_length"],
        output_dim=data_cfg["prediction_length"],
        conv_out_channels=model_cfg.get("conv_out_channels", 64),
        kernel_size=model_cfg.get("kernel_size", 3),
        hidden_dim1=model_cfg.get("hidden_dim1", 64),
        hidden_dim2=model_cfg.get("hidden_dim2", 8),
        activation=model_cfg.get("activation", "relu"),
        dropout=model_cfg.get("dropout", 0.1),
        use_adaptive_pool=model_cfg.get("use_adaptive_pool", False),
        lr=train_cfg.get("lr", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 1e-5),
        loss_name=train_cfg.get("loss_name", "quantile"),
        optimizer_name=train_cfg.get("optimizer_name", "adam"),
        quantiles=train_cfg.get("quantiles", [0.1, 0.5, 0.9]),
    )


def main(config_path: str) -> None:
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))

    seed_all(cfg["run"]["seed"])

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    run_dir = make_run_dir(cfg["run"]["reports_dir"], cfg["run"]["name"] + "_cnn_all")
    ckpt_dir = run_dir / "checkpoints"
    table_dir = run_dir / "tables"
    meta_dir = run_dir / "meta"

    ensure_dir(run_dir)
    ensure_dir(ckpt_dir)
    ensure_dir(table_dir)
    ensure_dir(meta_dir)
    copy_config(config_path, run_dir)

    # -----------------------
    # 1) DATA + SPLIT
    # -----------------------
    df_full = build_df_full(data_cfg)
    split = _resolve_split(cfg, df_full)

    print(
        f"[split] train={split['train_ratio']:.2f}, "
        f"val={split['val_ratio']:.2f}, test={split['test_ratio']:.2f} | "
        f"rows: train={split['train_rows']}, "
        f"val={split['val_rows']}, "
        f"test={split['test_rows']}"
    )
    print(
        f"[split] train_cutoff_row={split['train_cutoff']}, "
        f"val_cutoff_row={split['val_cutoff']}"
    )
    print(
        f"[split] train_cutoff_time_idx={split['train_cutoff_time_idx']}, "
        f"val_cutoff_time_idx={split['val_cutoff_time_idx']}, "
        f"test_start_time_idx={split['test_start_time_idx']}, "
        f"test_end_time_idx={split['test_end_time_idx']}"
    )

    # -----------------------
    # 2) DATAMODULE / MODEL
    # -----------------------
    dm = _build_datamodule(cfg, df_full, split)
    model = _build_model(cfg)

    # shape 확인용. 필요 없으면 이 블록은 지워도 됨.
    dm.setup("fit")
    xb, yb, meta = next(iter(dm.train_dataloader()))
    print("train x shape:", xb.shape)
    print("train y shape:", yb.shape)
    print("example meta:", meta)

    # -----------------------
    # 3) TRAIN + SAVE BEST CKPT
    # -----------------------
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
    )

    trainer.fit(model, datamodule=dm)

    best_ckpt = ckpt_cb.best_model_path or ckpt_cb.last_model_path
    if not best_ckpt:
        best_ckpt = str(ckpt_dir / "last.ckpt")
        trainer.save_checkpoint(best_ckpt)

    dm.setup("test")
    n_test_samples = len(dm.test_dataset)

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
        "n_test_samples": n_test_samples,
    }
    save_json(train_meta, meta_dir / "train_meta.json")

    # -----------------------
    # 4) LOAD BEST CKPT -> PREDICT
    # -----------------------
    best_model = CNN1DQuantileLightningModule.load_from_checkpoint(best_ckpt)

    pred_trainer = pl.Trainer(
        accelerator=train_cfg.get("accelerator", "auto"),
        devices=train_cfg.get("devices", 1),
        default_root_dir=str(run_dir),
        logger=False,
        enable_checkpointing=False,
    )

    dm.setup("test")
    test_loader = dm.test_dataloader()

    pred_outputs = pred_trainer.predict(best_model, dataloaders=test_loader)
    final_df, one = postprocess_cnn_predictions(
        pred_outputs,
        df_full,
        quantiles=train_cfg.get("quantiles", [0.1, 0.5, 0.9]),
    )

    if "horizon_idx" not in final_df.columns:
        raise ValueError("CNN prediction output is missing horizon_idx; multi-horizon postprocess failed.")

    expected_horizon = int(data_cfg["prediction_length"])
    actual_horizon = int(final_df["horizon_idx"].max())
    if actual_horizon != expected_horizon:
        raise ValueError(
            f"Config prediction_length={expected_horizon} does not match "
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

    # -----------------------
    # 5) EVAL
    # -----------------------
    eval_cnn_main(config_path, str(run_dir))

    print("RUN_DIR:", run_dir)
    print("BEST_CKPT:", best_ckpt)
    print("SAVED:", out_one)
    print("SAVED:", out_all)
    print("SAVED:", out_horizon_metrics)
    print("SAVED:", run_dir / "reports" / "metrics.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)
