"""Compare TFT interpretation tensors for two trained checkpoints."""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import yaml
from pytorch_forecasting import TimeSeriesDataSet

from pv_forecast.data.build_df_full import build_df_full
from pv_forecast.models.tft_vendor import (
    SeparateMambaTemporalFusionTransformer,
    TemporalFusionTransformer,
)
from scripts.run_all2 import _build_training_dataset, _resolve_known_unknown_reals
from scripts.run_all3 import add_cyclical_time_features

warnings.filterwarnings("ignore", message="X does not have valid feature names.*")


def load_inputs(run_dir: Path):
    config_path = next((run_dir / "configs").glob("*.yaml"))
    cfg = yaml.safe_load(config_path.read_text())
    meta = json.loads((run_dir / "meta" / "train_meta.json").read_text())
    df = build_df_full(cfg["data"])
    add_cyclical_time_features(df, cfg)
    known, unknown, _ = _resolve_known_unknown_reals(cfg, df)
    cutoff = int(len(df) * float(cfg["split"]["train_ratio"]))
    training = _build_training_dataset(df.iloc[:cutoff].copy(), cfg, known, unknown)
    test = TimeSeriesDataSet.from_dataset(
        training,
        df,
        min_prediction_idx=int(meta["val_cutoff_time_idx"]),
        stop_randomization=True,
    )
    loader = test.to_dataloader(train=False, batch_size=cfg["tft"]["batch_size"], num_workers=0)
    return cfg, meta, loader


def analyze(run_dir: Path, kind: str) -> dict:
    cfg, meta, loader = load_inputs(run_dir)
    cls = SeparateMambaTemporalFusionTransformer if kind == "mamba" else TemporalFusionTransformer
    # Some historical Mamba checkpoints retain unused LSTM-context weights
    # from the parent TFT implementation.
    model = cls.load_from_checkpoint(meta["best_ckpt"], map_location="cpu", strict=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    horizons = int(cfg["tft"]["max_prediction_length"])
    totals = None
    count = 0
    with torch.inference_mode():
        for x, _ in loader:
            x = {k: v.to(device) if torch.is_tensor(v) else v for k, v in x.items()}
            out = model(x)
            batch = []
            for horizon in range(horizons):
                batch.append(model.interpret_output(out, reduction="sum", attention_prediction_horizon=horizon))
            if totals is None:
                totals = {
                    "attention": [z["attention"].detach().cpu().double() for z in batch],
                    "encoder": batch[0]["encoder_variables"].detach().cpu().double(),
                    "decoder": batch[0]["decoder_variables"].detach().cpu().double(),
                    "static": batch[0]["static_variables"].detach().cpu().double(),
                }
            else:
                for h, z in enumerate(batch):
                    totals["attention"][h] += z["attention"].detach().cpu().double()
                totals["encoder"] += batch[0]["encoder_variables"].detach().cpu().double()
                totals["decoder"] += batch[0]["decoder_variables"].detach().cpu().double()
                totals["static"] += batch[0]["static_variables"].detach().cpu().double()
            count += int(x["encoder_lengths"].shape[0])

    def importance(values, names):
        values = values.numpy()
        values = values / values.sum() * 100
        return dict(sorted(zip(names, values.tolist()), key=lambda z: z[1], reverse=True))

    attention = []
    for values in totals["attention"]:
        # Later horizons additionally contain preceding decoder positions.
        # Restrict the cross-model comparison to the 77 encoder positions.
        values = values.numpy()[: int(cfg["tft"]["max_encoder_length"])]
        values = values / values.sum()
        attention.append(values.tolist())
    mean_attention = np.mean(np.asarray(attention), axis=0)
    mean_attention /= mean_attention.sum()
    recent = {
        "last_1_step_pct": float(mean_attention[-1] * 100),
        "last_11_steps_pct": float(mean_attention[-11:].sum() * 100),
        "last_22_steps_pct": float(mean_attention[-22:].sum() * 100),
        "last_44_steps_pct": float(mean_attention[-44:].sum() * 100),
        "oldest_33_steps_pct": float(mean_attention[:33].sum() * 100),
        "attention_center_lag": float(np.sum(mean_attention * np.arange(76, -1, -1))),
        "attention_entropy": float(-(mean_attention * np.log(mean_attention + 1e-12)).sum()),
    }
    return {
        "run_dir": str(run_dir),
        "checkpoint": meta["best_ckpt"],
        "samples": count,
        "encoder_variables_pct": importance(totals["encoder"], model.encoder_variables),
        "decoder_variables_pct": importance(totals["decoder"], model.decoder_variables),
        "static_variables_pct": importance(totals["static"], model.static_variables),
        "attention_summary": recent,
        "mean_attention_by_lag_pct": {
            str(lag): float(mean_attention[76 - lag] * 100) for lag in range(77)
        },
        "attention_by_horizon": attention,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tft-run", type=Path, required=True)
    parser.add_argument("--mamba-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "tft": analyze(args.tft_run, "tft"),
        "mamba_tft": analyze(args.mamba_run, "mamba"),
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
