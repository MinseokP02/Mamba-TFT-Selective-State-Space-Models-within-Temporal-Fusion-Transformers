from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import shlex
import subprocess
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelSpec:
    config: str
    command: str
    section_by_parameter: dict[str, str]
    metrics_paths: tuple[str, ...] = ("meta/metrics.json",)
    # One public parameter that sets several native config keys together.
    tied_keys: dict[str, tuple[str, ...]] = field(default_factory=dict)


# Public parameter names are deliberately shared where the models mean the same
# thing.  The value tells us which config section owns that parameter.
MODEL_SPECS: dict[str, ModelSpec] = {
    "lstm": ModelSpec(
        config="configs/lstm_utrecht_weather_single_id.yaml",
        command="all-lstm",
        section_by_parameter={
            "hidden_size": "lstm", "dropout": "lstm", "batch_size": "lstm",
            "learning_rate": "lstm", "num_layers": "lstm",
            "window_size": "lstm", "max_encoder_length": "lstm",
            "max_epochs": "lstm",
        },
    ),
    "cnn": ModelSpec(
        config="configs/cnn1d_utrecht_weather_single_id.yaml",
        command="all-cnn3",
        section_by_parameter={
            "hidden_size": "model",
            "dropout": "model", "conv_out_channels": "model",
            "kernel_size": "model", "hidden_dim1": "model",
            "hidden_dim2": "model", "batch_size": "train",
            "learning_rate": "train", "weight_decay": "train",
            "max_epochs": "train", "max_encoder_length": "data",
        },
        metrics_paths=("meta/metrics.json", "reports/metrics.json"),
        tied_keys={"hidden_size": ("conv_out_channels", "hidden_dim1", "hidden_dim2")},
    ),
    "simple_mamba": ModelSpec(
        config="configs/simple_mamba_utrecht_weather_single_id.yaml",
        command="all-simple-mamba",
        section_by_parameter={
            "hidden_size": "model", "dropout": "model",
            "batch_size": "train", "learning_rate": "train",
            "weight_decay": "train", "max_epochs": "train",
            "max_encoder_length": "model", "mamba_layers": "model",
            "mamba_d_state": "model", "mamba_d_conv": "model",
            "mamba_expand": "model",
        },
    ),
    "tft": ModelSpec(
        config="configs/tft_utrecht_weather_single_id.yaml",
        command="all3",
        section_by_parameter={
            "hidden_size": "tft", "dropout": "tft", "batch_size": "tft",
            "learning_rate": "tft", "attention_head_size": "tft",
            "lstm_layers": "tft", "max_encoder_length": "tft",
            "max_epochs": "tft", "gradient_clip_val": "tft",
        },
    ),
    "mamba_tft": ModelSpec(
        config="configs/mamba_tft_utrecht_weather_single_id.yaml",
        command="mt_all2",
        section_by_parameter={
            "hidden_size": "tft", "dropout": "tft", "batch_size": "tft",
            "learning_rate": "tft", "gradient_clip_val": "tft",
            "attention_head_size": "tft", "max_encoder_length": "tft",
            "max_epochs": "tft", "mamba_layers": "tft",
            "mamba_d_state": "tft", "mamba_d_conv": "tft",
            "mamba_expand": "tft",
        },
    ),
}

# Dataset-specific baseline configs. MODEL_SPECS keeps the Utrecht defaults for
# backward compatibility, while --dataset selects a complete comparable config
# set without requiring a custom --config for each model.
DATASET_CONFIGS: dict[str, dict[str, str]] = {
    "utrecht": {
        model: spec.config for model, spec in MODEL_SPECS.items()
    },
    "osong": {
        "lstm": "configs/lstm_run.yaml",
        "cnn": "configs/cnn1d_quantile.yaml",
        "simple_mamba": "configs/simple_mamba_all.yaml",
        "tft": "configs/tft_all.yaml",
        "mamba_tft": "configs/mamba_tft_all.yaml",
    },
}

PARAMETER_OPTIONS = {
    "hidden_size": "hidden-size", "dropout": "dropout",
    "batch_size": "batch-size", "learning_rate": "learning-rate",
    "gradient_clip_val": "max-gradient-norm",
    "attention_head_size": "attention-head-size", "lstm_layers": "lstm-layers",
    "num_layers": "num-layers", "window_size": "window-size",
    "max_encoder_length": "max-encoder-length", "max_epochs": "max-epochs",
    "weight_decay": "weight-decay", "conv_out_channels": "conv-out-channels",
    "kernel_size": "kernel-size", "hidden_dim1": "hidden-dim1",
    "hidden_dim2": "hidden-dim2", "mamba_layers": "mamba-layers",
    "mamba_d_state": "mamba-d-state", "mamba_d_conv": "mamba-d-conv",
    "mamba_expand": "mamba-expand",
}
INTEGER_PARAMETERS = {
    "hidden_size", "batch_size", "attention_head_size", "lstm_layers",
    "num_layers", "window_size", "max_encoder_length", "max_epochs",
    "conv_out_channels", "kernel_size", "hidden_dim1", "hidden_dim2",
    "mamba_layers", "mamba_d_state", "mamba_d_conv", "mamba_expand",
}
SUMMARY_METRICS = [
    "mae_p50", "rmse_p50", "coverage_10_90", "width_10_90",
    "qloss_q10", "qloss_q50", "qloss_q90",
]


def _parse_values(raw: str, *, integer: bool) -> list[int | float]:
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("At least one value is required.")
    values = [int(value) if integer else float(value) for value in parts]
    return list(dict.fromkeys(values))


def _same_value(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(float(left) - float(right)) < 1e-12
    return left == right


def _build_trials(
    baseline: dict[str, Any], candidates: dict[str, list[int | float]], mode: str,
) -> list[dict[str, Any]]:
    if mode == "one-factor":
        trials = [dict(baseline)]
        for parameter, values in candidates.items():
            for value in values:
                if not _same_value(value, baseline[parameter]):
                    trials.append({**baseline, parameter: value})
        return trials
    names = list(baseline)
    values = [candidates.get(name, [baseline[name]]) for name in names]
    return [dict(zip(names, combination)) for combination in itertools.product(*values)]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "model", "trial", "repeat", "seed", "status", *PARAMETER_OPTIONS,
        *SUMMARY_METRICS, "report_dir", "metrics_path", "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _run_trial(project_root: Path, command_name: str, config_path: Path) -> tuple[int, str | None, str | None]:
    command = [sys.executable, str(project_root / "run.py"), command_name, "--config", str(config_path)]
    print("\nCOMMAND:", shlex.join(command), flush=True)
    process = subprocess.Popen(
        command, cwd=project_root, env=os.environ.copy(), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    report_dir = None
    last_lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        stripped = line.strip()
        last_lines = (last_lines + [stripped])[-20:]
        if stripped.startswith("RUN_DIR:"):
            report_dir = stripped.split(":", 1)[1].strip()
    return_code = process.wait()
    return return_code, report_dir, None if return_code == 0 else "\n".join(last_lines)


def _parse_models(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(MODEL_SPECS)
    models = list(dict.fromkeys(part.strip().lower().replace("-", "_") for part in raw.split(",") if part.strip()))
    invalid = [model for model in models if model not in MODEL_SPECS]
    if invalid:
        raise ValueError(f"Unknown models: {invalid}. Choose from {list(MODEL_SPECS)} or all.")
    if not models:
        raise ValueError("--models did not contain a model.")
    return models


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--models", default="mamba_tft",
        help="Comma-separated: lstm,cnn,simple_mamba,tft,mamba_tft; or all.",
    )
    parser.add_argument("--config", help="Custom config (only valid when one model is selected).")
    parser.add_argument(
        "--dataset", choices=sorted(DATASET_CONFIGS), default="utrecht",
        help="Baseline config set to use. Ignored when --config is supplied.",
    )
    parser.add_argument("--mode", choices=["one-factor", "grid"], default="one-factor")
    parser.add_argument("--metric", default="mae_p50")
    parser.add_argument("--direction", choices=["min", "max"], default="min")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seeds", help="Comma-separated seeds. Overrides base seed for repeats.")
    parser.add_argument("--max-runs", type=int, default=100)
    parser.add_argument("--run", action="store_true", help="Start training; otherwise only print the plan.")
    for parameter, option in PARAMETER_OPTIONS.items():
        aliases = [f"--{option}"]
        underscore = f"--{parameter}"
        if underscore not in aliases:
            aliases.append(underscore)
        parser.add_argument(*aliases, dest=parameter, help="One value or comma-separated values.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Utrecht or Osong model ablations.")
    add_arguments(parser)
    return parser


def _config_path(
    project_root: Path, model: str, custom: str | None, dataset: str,
) -> Path:
    raw = custom or DATASET_CONFIGS[dataset][model]
    path = Path(raw)
    return path if path.is_absolute() else (project_root / path).resolve()


def _config_key(parameter: str, section: str) -> str:
    """Map a shared CLI parameter to each model's native config key."""
    if parameter == "learning_rate" and section == "train":
        return "lr"
    if parameter == "max_encoder_length" and section == "data":
        return "encoder_length"
    if parameter == "max_encoder_length" and section == "model":
        return "encoder_length"
    if parameter == "max_encoder_length" and section == "lstm":
        return "window_size"
    if section == "model":
        return {
            "hidden_size": "d_model",
            "mamba_layers": "num_mamba_layers",
            "mamba_d_state": "d_state",
            "mamba_d_conv": "d_conv",
            "mamba_expand": "expand",
        }.get(parameter, parameter)
    return parameter


def _native_keys(spec: ModelSpec, parameter: str, section: str) -> tuple[str, ...]:
    return spec.tied_keys.get(parameter) or (_config_key(parameter, section),)


def _baseline_and_candidates(cfg: dict[str, Any], spec: ModelSpec, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, list[int | float]]]:
    supplied = {name: getattr(args, name) for name in PARAMETER_OPTIONS if getattr(args, name) is not None}
    unsupported = [name for name in supplied if name not in spec.section_by_parameter]
    if unsupported:
        raise ValueError(f"Parameters not supported by selected model: {unsupported}")
    baseline: dict[str, Any] = {}
    candidates: dict[str, list[int | float]] = {}
    for parameter in supplied:
        section = spec.section_by_parameter[parameter]
        config_key = _native_keys(spec, parameter, section)[0]
        if config_key not in cfg.get(section, {}):
            raise KeyError(f"Missing {section}.{config_key} in config")
        baseline[parameter] = cfg[section][config_key]
        candidates[parameter] = _parse_values(supplied[parameter], integer=parameter in INTEGER_PARAMETERS)
    # No candidates means one baseline run using all tunable values present in the config.
    if not baseline:
        for parameter, section in spec.section_by_parameter.items():
            if parameter in spec.tied_keys:
                continue
            config_key = _config_key(parameter, section)
            if config_key in cfg.get(section, {}):
                baseline[parameter] = cfg[section][config_key]
    return baseline, candidates


def _apply_trial(cfg: dict[str, Any], spec: ModelSpec, trial: dict[str, Any]) -> None:
    for parameter, value in trial.items():
        section = spec.section_by_parameter[parameter]
        for config_key in _native_keys(spec, parameter, section):
            cfg[section][config_key] = value


def run_from_args(args: argparse.Namespace) -> None:
    if args.repeat < 1:
        raise ValueError(f"repeat must be >= 1, got {args.repeat}")
    models = _parse_models(args.models)
    if args.config and len(models) != 1:
        raise ValueError("--config can only be used with one selected model.")
    project_root = Path(__file__).resolve().parents[1]
    plans: list[tuple[str, ModelSpec, Path, dict[str, Any], list[dict[str, Any]], list[int]]] = []
    total_runs = 0
    for model in models:
        spec = MODEL_SPECS[model]
        config_path = _config_path(project_root, model, args.config, args.dataset)
        if not config_path.exists():
            raise FileNotFoundError(f"Config not found: {config_path}")
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        baseline, candidates = _baseline_and_candidates(cfg, spec, args)
        trials = _build_trials(baseline, candidates, args.mode)
        if args.seeds:
            seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
            if args.repeat != 1 and args.repeat != len(seeds):
                raise ValueError(f"repeat={args.repeat} does not match seeds={len(seeds)}")
        else:
            seeds = [int(cfg["run"].get("seed", 42)) + index for index in range(args.repeat)]
        total_runs += len(trials) * len(seeds)
        plans.append((model, spec, config_path, cfg, trials, seeds))

    print("Models:", ", ".join(models))
    print("Dataset:", args.dataset if not args.config else f"custom ({args.config})")
    print("Search mode:", args.mode)
    print("Total runs:", total_runs)
    print("Selection:", args.direction, args.metric)
    for model, _, config_path, _, trials, seeds in plans:
        print(f"\n[{model}] config={config_path}, combinations={len(trials)}, seeds={seeds}")
        for index, trial in enumerate(trials, 1):
            print(f"  [{index:03d}] " + ", ".join(f"{key}={value}" for key, value in trial.items()))
    if total_runs > args.max_runs:
        raise ValueError(f"Planned runs ({total_runs}) exceed --max-runs={args.max_runs}.")
    if not args.run:
        print("\nDRY RUN: add --run to start the experiments.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_root = Path(plans[0][3]["run"]["reports_dir"])
    if not reports_root.is_absolute():
        reports_root = project_root / reports_root
    dataset_label = args.dataset if not args.config else "custom"
    session_dir = reports_root / f"ablation_{timestamp}_{'_'.join(models)}_{dataset_label}"
    generated_dir = session_dir / "generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=False)
    (session_dir / "command.txt").write_text(shlex.join(sys.argv) + "\n", encoding="utf-8")
    rows: list[dict[str, Any]] = []
    run_number = 0
    for model, spec, _, base_cfg, trials, seeds in plans:
        for trial_index, trial in enumerate(trials, 1):
            for repeat_index, seed in enumerate(seeds, 1):
                run_number += 1
                trial_id = f"{model}_t{trial_index:04d}_r{repeat_index:02d}"
                cfg = deepcopy(base_cfg)
                _apply_trial(cfg, spec, trial)
                cfg["run"]["seed"] = seed
                cfg["run"]["name"] = f"{base_cfg['run']['name']}_{timestamp}_{trial_id}"
                cfg["run"]["artifacts_dir"] = str(session_dir / "artifacts")
                generated_config = generated_dir / f"{trial_id}.yaml"
                generated_config.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
                print(f"\n{'=' * 72}\nTRIAL {run_number}/{total_runs}: {trial_id}, seed={seed}\n{'=' * 72}", flush=True)
                return_code, report_raw, error = _run_trial(project_root, spec.command, generated_config)
                report_dir = (Path(report_raw) if report_raw else None)
                if report_dir is not None and not report_dir.is_absolute():
                    report_dir = project_root / report_dir
                metrics_path = None
                if report_dir:
                    metrics_path = next((report_dir / rel for rel in spec.metrics_paths if (report_dir / rel).exists()), None)
                metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if return_code == 0 and metrics_path else {}
                status = "completed" if metrics else "failed"
                if status == "failed" and error is None:
                    error = f"No readable metrics file in {spec.metrics_paths}."
                rows.append({
                    "model": model, "trial": trial_id, "repeat": repeat_index,
                    "seed": seed, "status": status, **trial,
                    **{metric: metrics.get(metric) for metric in SUMMARY_METRICS},
                    "report_dir": str(report_dir or ""), "metrics_path": str(metrics_path or ""),
                    "error": error or "",
                })
                _write_summary_csv(session_dir / "summary.csv", rows)
                _write_json(session_dir / "summary.json", rows)

    completed = [row for row in rows if row["status"] == "completed" and row.get(args.metric) is not None]
    if not completed:
        raise RuntimeError(f"No completed run contained metric={args.metric}. Summary: {session_dir / 'summary.csv'}")
    reverse = args.direction == "max"
    best_by_model = {
        model: sorted((row for row in completed if row["model"] == model), key=lambda row: float(row[args.metric]), reverse=reverse)[0]
        for model in models if any(row["model"] == model for row in completed)
    }
    best = sorted(completed, key=lambda row: float(row[args.metric]), reverse=reverse)[0]
    _write_json(session_dir / "best_by_model.json", best_by_model)
    _write_json(session_dir / "best_result.json", best)
    print(f"\nABLATION COMPLETE: completed={len(completed)}, failed={len(rows) - len(completed)}")
    for model, result in best_by_model.items():
        print(f"  {model}: best {args.metric}={result[args.metric]} ({result['report_dir']})")
    print("Summary:", session_dir / "summary.csv")


def main(argv: list[str] | None = None) -> None:
    run_from_args(_build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
