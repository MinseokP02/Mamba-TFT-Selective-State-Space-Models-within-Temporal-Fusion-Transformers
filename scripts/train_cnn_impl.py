# src/pv_forecast/scripts_like/train_cnn_impl.py

from __future__ import annotations

from pathlib import Path

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor

from pv_forecast.data.seq_dataset import SequenceDataModule
from pv_forecast.data.build_df_full import build_df_full
# from pv_forecast.models.cnn1d_module import CNN1DLightningModule
# from pv_forecast.models.cnn1d_module_paper import CNN1DLightningModule
from pv_forecast.models.cnn1d_quantile import CNN1DQuantileLightningModule
from pv_forecast.common.seed import seed_all
from pv_forecast.common.io import load_yaml, ensure_dir
from pv_forecast.common.paths import make_run_dir, copy_config, save_json


def train_cnn_main(config_path: str):
    cfg = load_yaml(config_path)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    seed_all(cfg.get("seed", 42))

    df_full = build_df_full(data_cfg)

    split_cfg = cfg.get("split", {})
    train_ratio = float(split_cfg.get("train_ratio", 0.7))
    val_ratio = float(split_cfg.get("val_ratio", 0.15))
    test_ratio = float(split_cfg.get("test_ratio", 0.15))

    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError(
            f"split ratios must sum to 1.0, "
            f"but got train={train_ratio}, val={val_ratio}, test={test_ratio}, sum={ratio_sum}"
        )
    
    time_idx_col = data_cfg.get("time_idx_col", "time_idx")
    time_indices = (
        df_full[time_idx_col]
        .drop_duplicates()
        .sort_values()
        .to_list()
    )

    n_time = len(time_indices)
    train_end = int(n_time * train_ratio)
    val_end = int(n_time * (train_ratio + val_ratio))

    if train_end <= 0 or train_end >= val_end or val_end >= n_time:
        raise ValueError(
            f"Invalid split for n_time={n_time}: "
            f"train_ratio={train_ratio}, val_ratio={val_ratio}, test_ratio={test_ratio}, "
            f"train_end={train_end}, val_end={val_end}"
        )
    
    training_cutoff = int(time_indices[train_end - 1])
    validation_cutoff = int(time_indices[val_end - 1])
    test_start_time_idx = int(time_indices[val_end])
    test_end_time_idx = int(time_indices[-1])
    cutoff_time_idx = training_cutoff

    train_rows = int((df_full[time_idx_col] <= training_cutoff).sum())
    val_rows = int(((df_full[time_idx_col] > training_cutoff) & (df_full[time_idx_col] <= validation_cutoff)).sum())
    test_rows = int((df_full[time_idx_col] > validation_cutoff).sum())

    print(
        f"[split] train={train_ratio:.2f}, val={val_ratio:.2f}, test={test_ratio:.2f} | "
        f"time steps: train={train_end}, val={val_end - train_end}, test={n_time - val_end} | "
        f"rows: train={train_rows}, val={val_rows}, test={test_rows}"
    )
    print(
        f"[split] training_cutoff={training_cutoff}, "
        f"validation_cutoff={validation_cutoff}, "
        f"test_start_time_idx={test_start_time_idx}, test_end_time_idx={test_end_time_idx}"
    )

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

    # model = CNN1DLightningModule(
    #     num_features=len(data_cfg["feature_cols"]),
    #     encoder_length=data_cfg["encoder_length"],
    #     output_dim=data_cfg["prediction_length"],
    #     conv_out_channels=model_cfg.get("hidden_channels", 64),
    #     kernel_size=model_cfg.get("kernel_size", 3),
    #     dropout=model_cfg.get("dropout", 0.1),
    #     lr=train_cfg.get("lr", 1e-3),
    #     weight_decay=train_cfg.get("weight_decay", 1e-5),
    #     loss_name=train_cfg.get("loss_name", "mae"),
    # )

    model = CNN1DQuantileLightningModule(
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


    dm.setup("fit")
    xb, yb, meta = next(iter(dm.train_dataloader()))
    print("train x shape:", xb.shape)
    print("train y shape:", yb.shape)
    print("example meta:", meta)

    run_dir = make_run_dir(cfg["run"]["reports_dir"], cfg["run"]["name"])
    ckpt_dir = run_dir / "checkpoints"
    ensure_dir(run_dir)
    ensure_dir(ckpt_dir)
    copy_config(config_path, run_dir)

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=train_cfg.get("early_stop_patience", 10),
            mode="min",
        ),
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="best-{epoch:02d}-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = pl.Trainer(
        max_epochs=train_cfg.get("max_epochs", 50),
        accelerator=train_cfg.get("accelerator", "auto"),
        devices=train_cfg.get("devices", 1),
        callbacks=callbacks,
        log_every_n_steps=train_cfg.get("log_every_n_steps", 10),
        default_root_dir=str(run_dir),
    )

    trainer.fit(model, datamodule=dm)

    best_ckpt = callbacks[1].best_model_path

    ensure_dir(run_dir / "meta")
    save_json(
        {
            "run_dir": str(run_dir),
            "best_ckpt": best_ckpt,
            "n_total": n_time,
            "train_cutoff": training_cutoff,
            "val_cutoff": validation_cutoff,
            "train_cutoff_time_idx": training_cutoff,
            "val_cutoff_time_idx": validation_cutoff,
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "n_train_rows": train_end,
            "n_val_samples": val_end - train_end,
            "n_test_samples": n_time - val_end,
        },
        run_dir / "meta" / "train_meta.json"
    )

    print("RUN_DIR:", run_dir)
    print("BEST_CKPT:", best_ckpt)
    print(f"[CNN1D][train] best_ckpt={best_ckpt}")