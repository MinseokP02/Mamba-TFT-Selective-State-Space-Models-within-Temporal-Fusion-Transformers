# src/pv_forecast/scripts_like/predict_cnn_impl.py

from __future__ import annotations

from pathlib import Path

import lightning.pytorch as pl
import yaml

from pv_forecast.data.build_df_full import build_df_full
from pv_forecast.data.seq_dataset import SequenceDataModule
from pv_forecast.models.cnn1d_module_paper import CNN1DLightningModule
from pv_forecast.models.cnn1d_quantile import CNN1DQuantileLightningModule
from pv_forecast.common.io import (
    ensure_dir,
    predictions_to_dataframe,
    postprocess_cnn_predictions,
)
from pv_forecast.common.paths import make_run_dir, copy_config, save_json


def predict_cnn_main(config_path: str, ckpt_path: str):
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))

    run_dir = make_run_dir(cfg["run"]["reports_dir"], cfg["run"]["name"] + "_pred")
    copy_config(config_path, run_dir)
    save_json({"ckpt": ckpt_path}, run_dir / "meta" / "predict_meta.json")

    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    pred_cfg = cfg.get("predict", {})

    df_full = build_df_full(data_cfg)

    train_ratio = cfg["split"]["train_ratio"]

    n = len(df_full)
    train_end = int(n * train_ratio)
    val_end = n

    if train_end <= 0 or train_end >= n:
        raise ValueError(f"Invalid train_ratio={train_ratio} for n={n}")

    training_cutoff = int(df_full.iloc[train_end - 1]["time_idx"])
    validation_cutoff = int(df_full.iloc[val_end - 1]["time_idx"])

    dm = SequenceDataModule(
        df_full=df_full,
        feature_cols=data_cfg["feature_cols"],
        target_col=data_cfg["target_col"],
        group_col=data_cfg.get("group_col", "series_id"),
        time_idx_col=data_cfg.get("time_idx_col", "time_idx"),
        encoder_length=data_cfg["encoder_length"],
        prediction_length=data_cfg["prediction_length"],
        training_cutoff=training_cutoff,
        validation_cutoff=validation_cutoff,
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg.get("num_workers", 0),
        allow_missing_time_idx=data_cfg.get("allow_missing_time_idx", False),
        model_type="cnn1d",
    )

    model = CNN1DQuantileLightningModule.load_from_checkpoint(ckpt_path)

    trainer = pl.Trainer(
        accelerator=train_cfg.get("accelerator", "auto"),
        devices=train_cfg.get("devices", 1),
    )

    pred_outputs = trainer.predict(model, datamodule=dm)
    final_df, one = postprocess_cnn_predictions(pred_outputs, df_full, quantiles=train_cfg.get("quantiles", [0.1, 0.5, 0.9]))

    (run_dir / "tables").mkdir(parents=True, exist_ok=True)

    out_one = run_dir / "tables" / "pred_one.csv"
    out_all = run_dir / "tables" / "pred_all_samples.csv"

    one.to_csv(out_one, index=False, encoding="utf-8-sig")
    final_df.to_csv(out_all, index=False, encoding="utf-8-sig")