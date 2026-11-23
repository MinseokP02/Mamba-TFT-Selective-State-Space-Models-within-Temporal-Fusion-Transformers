"""Aggregate the unified hidden/dropout/lr grid ablation (report2).

usage: python analysis/grid_ablation_report.py <session_dir> [<session_dir> ...]
Writes reports/report2_per_run.csv and prints markdown tables to stdout.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ORDER = ["cnn", "lstm", "tft", "simple_mamba", "mamba_tft"]
LABEL = {"cnn": "CNN", "lstm": "LSTM", "tft": "TFT", "simple_mamba": "Mamba", "mamba_tft": "Mamba-TFT"}
PARAMS = ["hidden_size", "dropout", "learning_rate"]


def load(dirs: list[str]) -> pd.DataFrame:
    frames = [pd.read_csv(Path(d) / "summary.csv") for d in dirs if (Path(d) / "summary.csv").exists()]
    df = pd.concat(frames, ignore_index=True)
    df["hidden_size"] = df["hidden_size"].fillna(df.get("conv_out_channels"))
    return df


def fmt(mean: float, std: float) -> str:
    return f"{mean:.2f} ± {std:.2f}"


def md(df, index: bool = True) -> str:
    """Minimal markdown table (avoids the optional tabulate dependency)."""
    if isinstance(df, pd.Series):
        df = df.to_frame()
    d = df.reset_index() if index else df
    cols = [str(c) for c in d.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for row in d.itertuples(index=False):
        out.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in row) + " |")
    return "\n".join(out)


def main(dirs: list[str]) -> None:
    df = load(dirs)
    df.to_csv(ROOT / "reports" / "report2_per_run.csv", index=False)
    ok = df[df.status == "completed"].copy()
    print(f"runs={len(df)} completed={len(ok)} failed={len(df) - len(ok)}\n")

    combo = (
        ok.groupby(["model", *PARAMS])
        .agg(mae=("mae_p50", "mean"), mae_sd=("mae_p50", "std"), rmse=("rmse_p50", "mean"),
             cov=("coverage_10_90", "mean"), width=("width_10_90", "mean"), n=("seed", "nunique"))
        .reset_index()
    )
    combo.to_csv(ROOT / "reports" / "report2_per_combo.csv", index=False)

    print("## Best combo per model (mean over 5 seeds)\n")
    rows = []
    for m in ORDER:
        c = combo[combo.model == m]
        if c.empty:
            continue
        b = c.sort_values("mae").iloc[0]
        rows.append({"model": LABEL[m], "hidden": int(b.hidden_size), "dropout": b.dropout, "lr": b.learning_rate,
                     "MAE": fmt(b.mae, b.mae_sd), "RMSE": round(b.rmse, 2), "Cov": round(b["cov"], 3),
                     "Width": round(b.width, 1)})
    print(md(pd.DataFrame(rows), index=False), "\n")

    print("## Grid-average per model (all 27 combos)\n")
    g = ok.groupby("model").agg(mae=("mae_p50", "mean"), med=("mae_p50", "median"), best=("mae_p50", "min"),
                                worst=("mae_p50", "max")).reindex(ORDER).dropna(how="all")
    g.index = [LABEL[i] for i in g.index]
    print(md(g.round(2)), "\n")

    print("## Main effects (MAE mean over seeds and other params)\n")
    for p in PARAMS:
        t = ok.pivot_table(index="model", columns=p, values="mae_p50", aggfunc="mean").reindex(ORDER).dropna(how="all")
        t.index = [LABEL[i] for i in t.index]
        print(f"### {p}\n\n{md(t.round(2))}\n")

    print("## Sensitivity: share of combo-level MAE variance explained (one-way, %)\n")
    rows = []
    for m in ORDER:
        c = combo[combo.model == m]
        if c.empty:
            continue
        tot = ((c.mae - c.mae.mean()) ** 2).sum()
        r = {"model": LABEL[m]}
        for p in PARAMS:
            means = c.groupby(p).mae.transform("mean")
            r[p] = round(100 * ((means - c.mae.mean()) ** 2).sum() / tot, 1) if tot > 0 else 0
        seed_sd = ok[ok.model == m].groupby(PARAMS).mae_p50.std().mean()
        r["mean seed SD"] = round(seed_sd, 2)
        rows.append(r)
    print(md(pd.DataFrame(rows), index=False), "\n")

    print("## dropout x lr (MAE, averaged over hidden & seeds)\n")
    for m in ORDER:
        c = ok[ok.model == m]
        if c.empty:
            continue
        t = c.pivot_table(index="dropout", columns="learning_rate", values="mae_p50", aggfunc="mean")
        print(f"### {LABEL[m]}\n\n" + md(t.round(2)) + "\n")

    print("## hidden x lr (MAE, averaged over dropout & seeds)\n")
    for m in ORDER:
        c = ok[ok.model == m]
        if c.empty:
            continue
        t = c.pivot_table(index="hidden_size", columns="learning_rate", values="mae_p50", aggfunc="mean")
        print(f"### {LABEL[m]}\n\n" + md(t.round(2)) + "\n")

    print("## Model ranking per combo (rank by MAE)\n")
    piv = combo.pivot_table(index=PARAMS, columns="model", values="mae").reindex(columns=ORDER).dropna(axis=1, how="all")
    wins = piv.idxmin(axis=1).map(LABEL).value_counts()
    print(md(wins), "\n")
    piv.columns = [LABEL[c] for c in piv.columns]
    print(md(piv.round(2)), "\n")


if __name__ == "__main__":
    main(sys.argv[1:])
