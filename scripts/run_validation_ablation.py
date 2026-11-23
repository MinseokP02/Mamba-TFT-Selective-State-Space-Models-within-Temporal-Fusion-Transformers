"""Leakage-safe TFT/Mamba-TFT ablation using validation quantile loss.

Search trials never evaluate the test split. Hyperparameters are ranked by the
mean best validation loss across seeds. With ``--run-final-test``, only the
selected configuration is retrained and evaluated on test for each seed.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any

import yaml

from scripts.run_mt_ablation import (
    MODEL_SPECS,
    PARAMETER_OPTIONS,
    SUMMARY_METRICS,
    _apply_trial,
    _baseline_and_candidates,
    _build_trials,
    _config_path,
    _parse_models,
    _run_trial,
)


SUPPORTED_MODELS = {"tft", "mamba_tft"}


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], extra_metrics: list[str]) -> None:
    fields = [
        "model", "candidate", "trial", "repeat", "seed", "status",
        *PARAMETER_OPTIONS, *extra_metrics, "report_dir", "metrics_path", "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--models", default="mamba_tft",
        help="Comma-separated: tft,mamba_tft; or all.",
    )
    parser.add_argument("--config", help="Custom config; valid with one model only.")
    parser.add_argument("--mode", choices=["one-factor", "grid"], default="one-factor")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--seeds", help="Comma-separated seeds; overrides generated repeat seeds.")
    parser.add_argument("--max-runs", type=int, default=100)
    parser.add_argument("--run", action="store_true", help="Run validation-only search trials.")
    parser.add_argument(
        "--run-final-test", action="store_true",
        help="After selection, retrain/test only the best configuration for every seed.",
    )
    for parameter, option in PARAMETER_OPTIONS.items():
        aliases = [f"--{option}"]
        underscore = f"--{parameter}"
        if underscore not in aliases:
            aliases.append(underscore)
        parser.add_argument(*aliases, dest=parameter)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validation-only ablation followed by an optional final test.",
    )
    add_arguments(parser)
    return parser


def _seeds(cfg: dict[str, Any], args: argparse.Namespace) -> list[int]:
    if args.repeat < 1:
        raise ValueError("--repeat must be at least 1")
    if args.seeds:
        values = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
        if args.repeat not in {1, len(values)}:
            raise ValueError(f"repeat={args.repeat} does not match {len(values)} explicit seeds")
        return values
    base = int(cfg["run"].get("seed", 42))
    return [base + index for index in range(args.repeat)]


def _plans(args: argparse.Namespace, project_root: Path):
    models = _parse_models(args.models)
    if args.models.strip().lower() == "all":
        models = ["tft", "mamba_tft"]
    invalid = sorted(set(models) - SUPPORTED_MODELS)
    if invalid:
        raise ValueError(f"validation-ablation supports only {sorted(SUPPORTED_MODELS)}; got {invalid}")
    if args.config and len(models) != 1:
        raise ValueError("--config can only be used with one selected model")
    plans = []
    for model in models:
        spec = MODEL_SPECS[model]
        config_path = _config_path(project_root, model, args.config)
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        baseline, candidates = _baseline_and_candidates(cfg, spec, args)
        plans.append((model, spec, cfg, _build_trials(baseline, candidates, args.mode), _seeds(cfg, args)))
    return plans


def _aggregate(
    rows: list[dict[str, Any]], metric: str, expected_repeats: dict[str, int],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row["status"] == "completed" and row.get(metric) is not None:
            groups.setdefault((row["model"], row["candidate"]), []).append(row)
    aggregated = []
    for (model, candidate), members in groups.items():
        if len(members) != expected_repeats[model]:
            continue
        values = [float(row[metric]) for row in members]
        first = members[0]
        aggregated.append({
            "model": model,
            "candidate": candidate,
            "completed_repeats": len(members),
            "seeds": [row["seed"] for row in members],
            f"mean_{metric}": fmean(values),
            f"std_{metric}": pstdev(values) if len(values) > 1 else 0.0,
            "parameters": {name: first[name] for name in PARAMETER_OPTIONS if name in first},
            "source_config": first["generated_config"],
        })
    return aggregated


def run_from_args(args: argparse.Namespace) -> None:
    project_root = Path(__file__).resolve().parents[1]
    plans = _plans(args, project_root)
    total_runs = sum(len(trials) * len(seeds) for _, _, _, trials, seeds in plans)
    print("Selection: minimum mean validation quantile loss (val_loss)")
    print("Search test evaluation: disabled")
    print("Total validation runs:", total_runs)
    for model, _, _, trials, seeds in plans:
        print(f"[{model}] candidates={len(trials)}, seeds={seeds}")
        for index, trial in enumerate(trials, 1):
            print(f"  [{index:04d}] " + ", ".join(f"{key}={value}" for key, value in trial.items()))
    if total_runs > args.max_runs:
        raise ValueError(f"Planned runs ({total_runs}) exceed --max-runs={args.max_runs}")
    if not args.run:
        print("DRY RUN: add --run to start validation-only search")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    reports_root = Path(plans[0][2]["run"]["reports_dir"])
    if not reports_root.is_absolute():
        reports_root = project_root / reports_root
    session_dir = reports_root / f"validation_ablation_{timestamp}_{'_'.join(p[0] for p in plans)}_utrecht"
    generated_dir = session_dir / "generated_configs"
    generated_dir.mkdir(parents=True, exist_ok=False)
    (session_dir / "command.txt").write_text(shlex.join(sys.argv) + "\n", encoding="utf-8")

    rows: list[dict[str, Any]] = []
    for model, spec, base_cfg, trials, seeds in plans:
        for trial_index, trial in enumerate(trials, 1):
            candidate = f"{model}_t{trial_index:04d}"
            for repeat_index, seed in enumerate(seeds, 1):
                trial_id = f"{candidate}_r{repeat_index:02d}"
                cfg = deepcopy(base_cfg)
                _apply_trial(cfg, spec, trial)
                cfg["run"]["seed"] = seed
                cfg["run"]["name"] = f"{base_cfg['run']['name']}_{timestamp}_{trial_id}"
                cfg["run"]["artifacts_dir"] = str(session_dir / "artifacts")
                cfg["run"]["evaluate_test"] = False
                generated_config = generated_dir / f"{trial_id}.yaml"
                generated_config.write_text(
                    yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8",
                )
                print(f"\nVALIDATION TRIAL: {trial_id}, seed={seed}", flush=True)
                code, report_raw, error = _run_trial(project_root, spec.command, generated_config)
                report_dir = Path(report_raw) if report_raw else None
                if report_dir is not None and not report_dir.is_absolute():
                    report_dir = project_root / report_dir
                metrics_path = report_dir / "meta" / "selection_metrics.json" if report_dir else None
                metrics = (
                    json.loads(metrics_path.read_text(encoding="utf-8"))
                    if code == 0 and metrics_path and metrics_path.exists() else {}
                )
                status = "completed" if metrics.get("val_loss") is not None else "failed"
                if status == "failed" and error is None:
                    error = "No readable validation metric at meta/selection_metrics.json"
                rows.append({
                    "model": model, "candidate": candidate, "trial": trial_id,
                    "repeat": repeat_index, "seed": seed, "status": status, **trial,
                    "val_loss": metrics.get("val_loss"), "best_ckpt": metrics.get("best_ckpt"),
                    "generated_config": str(generated_config),
                    "report_dir": str(report_dir or ""), "metrics_path": str(metrics_path or ""),
                    "error": error or "",
                })
                _write_csv(session_dir / "validation_summary.csv", rows, ["val_loss", "best_ckpt"])
                _write_json(session_dir / "validation_summary.json", rows)

    expected_repeats = {model: len(seeds) for model, _, _, _, seeds in plans}
    aggregated = _aggregate(rows, "val_loss", expected_repeats)
    if not aggregated:
        raise RuntimeError("No candidate completed every required seed with a valid val_loss")
    _write_json(session_dir / "candidate_summary.json", aggregated)
    best_by_model = {}
    for model, *_ in plans:
        choices = [row for row in aggregated if row["model"] == model]
        best_by_model[model] = min(choices, key=lambda row: row["mean_val_loss"])
    _write_json(session_dir / "best_by_model.json", best_by_model)

    best_configs: dict[str, Path] = {}
    for model, best in best_by_model.items():
        cfg = yaml.safe_load(Path(best["source_config"]).read_text(encoding="utf-8"))
        cfg["run"]["evaluate_test"] = True
        cfg["run"]["name"] = f"{cfg['run']['name'].rsplit('_r', 1)[0]}_selected"
        path = session_dir / f"best_{model}.yaml"
        path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        best_configs[model] = path
        print(f"BEST {model}: mean_val_loss={best['mean_val_loss']:.6f} config={path}")

    if args.run_final_test:
        final_rows = []
        for model, spec, _, _, seeds in plans:
            selected = yaml.safe_load(best_configs[model].read_text(encoding="utf-8"))
            for repeat_index, seed in enumerate(seeds, 1):
                cfg = deepcopy(selected)
                cfg["run"]["seed"] = seed
                cfg["run"]["evaluate_test"] = True
                cfg["run"]["name"] = f"{selected['run']['name']}_final_r{repeat_index:02d}"
                path = session_dir / f"final_{model}_r{repeat_index:02d}.yaml"
                path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
                print(f"\nFINAL TEST: {model}, seed={seed}", flush=True)
                code, report_raw, error = _run_trial(project_root, spec.command, path)
                report_dir = Path(report_raw) if report_raw else None
                if report_dir is not None and not report_dir.is_absolute():
                    report_dir = project_root / report_dir
                metrics_path = report_dir / "meta" / "metrics.json" if report_dir else None
                metrics = (
                    json.loads(metrics_path.read_text(encoding="utf-8"))
                    if code == 0 and metrics_path and metrics_path.exists() else {}
                )
                final_rows.append({
                    "model": model, "candidate": best_by_model[model]["candidate"],
                    "trial": f"final_r{repeat_index:02d}", "repeat": repeat_index,
                    "seed": seed, "status": "completed" if metrics else "failed",
                    **{metric: metrics.get(metric) for metric in SUMMARY_METRICS},
                    "report_dir": str(report_dir or ""), "metrics_path": str(metrics_path or ""),
                    "error": error or "",
                })
                _write_csv(session_dir / "final_test_summary.csv", final_rows, SUMMARY_METRICS)
                _write_json(session_dir / "final_test_summary.json", final_rows)
        final_aggregate = {}
        for model, *_ in plans:
            members = [r for r in final_rows if r["model"] == model and r["status"] == "completed"]
            final_aggregate[model] = {
                metric: {
                    "mean": fmean(float(r[metric]) for r in members),
                    "std": pstdev(float(r[metric]) for r in members) if len(members) > 1 else 0.0,
                }
                for metric in SUMMARY_METRICS if members and all(r.get(metric) is not None for r in members)
            }
        _write_json(session_dir / "final_test_aggregate.json", final_aggregate)

    print("SESSION_DIR:", session_dir)


def main(argv: list[str] | None = None) -> None:
    run_from_args(_parser().parse_args(argv))


if __name__ == "__main__":
    main()
