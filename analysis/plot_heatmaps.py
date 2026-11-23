# src/pv_forecast/analysis/plot_heatmaps.py

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def load_prediction_csv(path: str | Path, label: str) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)

    required_cols = [
        "timestamp",
        "hour",
        "true",
        "pred_q10",
        "pred_q50",
        "pred_q90",
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"필수 컬럼이 없어. missing={missing}, 현재 컬럼={list(df.columns)}"
        )

    # timestamp가 정상적으로 있으면 timestamp 기준으로 처리
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"] = pd.to_datetime(df["timestamp"]).dt.floor("D")

    # timestamp 파싱이 실패하는 경우 date + hour로 복구
    if ts.isna().all():
        date = pd.to_datetime(df["date"], errors="coerce")
        hour = pd.to_numeric(df["hour"], errors="coerce").fillna(0).astype(int)
        ts = date + pd.to_timedelta(hour, unit="h")

    y_true = pd.to_numeric(df["true"], errors="coerce")
    q10 = pd.to_numeric(df["pred_q10"], errors="coerce")
    q50 = pd.to_numeric(df["pred_q50"], errors="coerce")
    q90 = pd.to_numeric(df["pred_q90"], errors="coerce")

    # q10 > q90인 quantile crossing이 혹시 있어도 coverage 계산이 깨지지 않게 처리
    lower = np.minimum(q10, q90)
    upper = np.maximum(q10, q90)

    out = pd.DataFrame(
        {
            "model": label,
            "timestamp": ts,
            "date": pd.to_datetime(df["date"], errors="coerce"),
            "hour": pd.to_numeric(df["hour"], errors="coerce"),
            "y_true": y_true,
            "q10": q10,
            "q50": q50,
            "q90": q90,
            "lower": lower,
            "upper": upper,
        }
    )

    # sample 컬럼이 있으면 보존
    if "sample" in df.columns:
        out["sample"] = df["sample"]

    # abs_err가 이미 있으면 그대로 사용하고, 없으면 새로 계산
    if "abs_err" in df.columns:
        out["abs_error"] = pd.to_numeric(df["abs_err"], errors="coerce")
    else:
        out["abs_error"] = (out["y_true"] - out["q50"]).abs()

    # sq_err도 있으면 보존
    if "sq_err" in df.columns:
        out["sq_error"] = pd.to_numeric(df["sq_err"], errors="coerce")
    else:
        out["sq_error"] = (out["y_true"] - out["q50"]) ** 2

    out = out.dropna(
        subset=["timestamp", "hour", "y_true", "q50", "lower", "upper", "abs_error"]
    ).copy()

    out["hour"] = out["hour"].astype(int)
    out["month"] = out["timestamp"].dt.month

    out["covered"] = (
        (out["y_true"] >= out["lower"]) &
        (out["y_true"] <= out["upper"])
    ).astype(float)

    out["width"] = out["upper"] - out["lower"]
    out["quantile_crossed"] = (out["q10"] > out["q90"]).astype(float)

    return out


def load_horizon_prediction_csv(path: str | Path, label: str) -> pd.DataFrame:
    """Load every forecast-origin/horizon pair for horizon-wise evaluation."""
    path = Path(path)
    df = pd.read_csv(path)

    horizon_col = next(
        (column for column in ("horizon_idx", "horizon") if column in df.columns),
        None,
    )
    required = ["true", "pred_q50"]
    missing = [column for column in required if column not in df.columns]
    if horizon_col is None:
        missing.append("horizon_idx or horizon")
    if missing:
        raise ValueError(
            f"horizon-wise MAE 계산에 필요한 컬럼이 없어. "
            f"path={path}, missing={missing}, columns={list(df.columns)}"
        )

    out = pd.DataFrame(
        {
            "model": label,
            "horizon": pd.to_numeric(df[horizon_col], errors="coerce"),
            "y_true": pd.to_numeric(df["true"], errors="coerce"),
            "q50": pd.to_numeric(df["pred_q50"], errors="coerce"),
        }
    ).dropna()
    out["horizon"] = out["horizon"].astype(int)
    out["abs_error"] = (out["y_true"] - out["q50"]).abs()
    return out


def build_horizon_wise_mae_figure(
    horizon_df: pd.DataFrame,
    out_dir: Path,
    model_order: list[str],
    file_stem: str = "fig9_horizon_wise_mae",
    title: str = "Horizon-wise forecasting error",
) -> None:
    """Plot q50 MAE separately for every forecasting horizon."""
    summary = (
        horizon_df.groupby(["model", "horizon"], observed=True)
        .agg(mae=("abs_error", "mean"), n=("abs_error", "size"))
        .reset_index()
    )
    summary["model"] = pd.Categorical(
        summary["model"], categories=model_order, ordered=True
    )
    summary = summary.sort_values(["model", "horizon"])
    summary.to_csv(
        out_dir / f"{file_stem}.csv", index=False, encoding="utf-8-sig"
    )

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for model_name in model_order:
        model_data = summary[summary["model"] == model_name]
        if model_data.empty:
            continue
        ax.plot(
            model_data["horizon"],
            model_data["mae"],
            marker="o",
            linewidth=1.8,
            label=model_name,
        )

    horizons = sorted(summary["horizon"].unique())
    ax.set_xticks(horizons)
    ax.set_xlabel("Forecasting horizon (hour ahead)")
    ax.set_ylabel("MAE of q50 prediction")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_figure(fig, out_dir / f"{file_stem}.png")

def plot_heatmap(
    matrix: pd.DataFrame,
    title: str,
    xlabel: str,
    ylabel: str,
    colorbar_label: str,
    out_path: Path,
    value_format: str = ".2f",
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    matrix = matrix.copy()

    n_rows, n_cols = matrix.shape
    fig_w = max(8.0, 0.6 * n_cols + 2.0)
    fig_h = max(3.5, 0.45 * n_rows + 2.0)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    values = matrix.to_numpy(dtype=float)
    im = ax.imshow(values, aspect="auto", vmin=vmin, vmax=vmax)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_xticks(np.arange(n_cols))
    ax.set_xticklabels(matrix.columns)

    ax.set_yticks(np.arange(n_rows))
    ax.set_yticklabels(matrix.index)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(colorbar_label)

    # 셀이 너무 많지 않을 때만 숫자 표시
    if n_rows * n_cols <= 160:
        for i in range(n_rows):
            for j in range(n_cols):
                val = values[i, j]
                if np.isfinite(val):
                    ax.text(
                        j,
                        i,
                        f"{val:{value_format}}",
                        ha="center",
                        va="center",
                        fontsize=8,
                    )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def build_hourly_coverage_heatmap(
    df: pd.DataFrame,
    out_dir: Path,
    start_hour: int,
    end_hour: int,
    nominal_coverage: float = 0.8,
) -> None:
    hours = list(range(start_hour, end_hour + 1))

    coverage = (
        df[df["hour"].between(start_hour, end_hour)]
        .groupby(["model", "hour"], observed=True)["covered"]
        .mean()
        .unstack("hour")
        .reindex(columns=hours)
    )

    coverage.to_csv(out_dir / "hourly_coverage_heatmap.csv", encoding="utf-8-sig")

    plot_heatmap(
        matrix=coverage,
        title="Hourly empirical coverage of q10-q90 interval",
        xlabel="Hour of day",
        ylabel="Model / Configuration",
        colorbar_label="Coverage",
        out_path=out_dir / "hourly_coverage_heatmap.png",
        value_format=".3f",
        vmin=0.0,
        vmax=1.0,
    )

    # 논문 해석용: nominal 0.8에서 얼마나 벗어났는지도 같이 저장
    calibration_error = (coverage - nominal_coverage).abs()
    calibration_error.to_csv(
        out_dir / "hourly_coverage_abs_error_from_0.8.csv",
        encoding="utf-8-sig",
    )

    plot_heatmap(
        matrix=calibration_error,
        title="Absolute deviation from nominal 80% coverage",
        xlabel="Hour of day",
        ylabel="Model / Configuration",
        colorbar_label="|Coverage - 0.8|",
        out_path=out_dir / "hourly_coverage_calibration_error_heatmap.png",
        value_format=".3f",
        vmin=0.0,
        vmax=0.5,
    )


def build_month_hour_mae_heatmap(
    df: pd.DataFrame,
    out_dir: Path,
    start_hour: int,
    end_hour: int,
) -> None:
    hours = list(range(start_hour, end_hour + 1))
    months = list(range(1, 13))

    for model_name, sub in df.groupby("model", observed=True):
        mae_matrix = (
            sub[sub["hour"].between(start_hour, end_hour)]
            .groupby(["month", "hour"], observed=True)["abs_error"]
            .mean()
            .unstack("hour")
            .reindex(index=months, columns=hours)
        )

        # 해당 데이터에 없는 월은 제거
        mae_matrix = mae_matrix.dropna(axis=0, how="all")

        file_name = safe_name(model_name)
        mae_matrix.to_csv(
            out_dir / f"month_hour_mae_heatmap_{file_name}.csv",
            encoding="utf-8-sig",
        )

        plot_heatmap(
            matrix=mae_matrix,
            title=f"Month-hour MAE heatmap ({model_name})",
            xlabel="Hour of day",
            ylabel="Month",
            colorbar_label="MAE",
            out_path=out_dir / f"month_hour_mae_heatmap_{file_name}.png",
            value_format=".1f",
        )

def save_figure(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def compute_model_metrics(df: pd.DataFrame, model_order: list[str]) -> pd.DataFrame:
    rows = []

    q_info = [
        (0.1, "q10"),
        (0.5, "q50"),
        (0.9, "q90"),
    ]

    for model_name, sub in df.groupby("model", observed=True):
        sub = sub.copy()
        denom = sub["y_true"].abs().sum()

        row = {
            "model": model_name,
            "n": len(sub),
            "mae_p50": sub["abs_error"].mean(),
            "rmse_p50": np.sqrt(sub["sq_error"].mean()),
            "coverage_10_90": sub["covered"].mean(),
            "width_10_90": sub["width"].mean(),
            "quantile_crossing_rate": sub["quantile_crossed"].mean(),
        }

        for q, col in q_info:
            err = sub["y_true"] - sub[col]
            pinball = np.maximum(q * err, (q - 1) * err)

            row[f"qloss_{col}"] = pinball.mean()
            row[f"qrisk_{col}"] = (
                2.0 * pinball.sum() / denom if denom > 0 else np.nan
            )

        rows.append(row)

    summary = pd.DataFrame(rows)

    summary["model"] = pd.Categorical(
        summary["model"],
        categories=model_order,
        ordered=True,
    )

    summary = summary.sort_values("model").reset_index(drop=True)

    return summary


def build_overall_performance_figure(
    summary: pd.DataFrame,
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    models = summary["model"].astype(str).tolist()
    x = np.arange(len(models))

    # (a) MAE / RMSE
    ax = axes[0]
    metrics = [
        ("mae_p50", "MAE"),
        ("rmse_p50", "RMSE"),
    ]
    bar_width = 0.35

    for i, (metric_col, metric_label) in enumerate(metrics):
        offset = (i - 0.5) * bar_width
        ax.bar(
            x + offset,
            summary[metric_col].to_numpy(),
            width=bar_width,
            label=metric_label,
        )

    ax.set_title("(a) Point forecasting error")
    ax.set_xlabel("Model")
    ax.set_ylabel("Error")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # (b) QRisk q50
    ax = axes[1]
    ax.bar(
        x,
        summary["qrisk_q50"].to_numpy(),
        width=0.55,
        label="QRisk q50",
    )

    ax.set_title("(b) Median quantile risk")
    ax.set_xlabel("Model")
    ax.set_ylabel("QRisk q50")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Overall performance comparison", y=1.03)
    fig.tight_layout()

    save_figure(fig, out_dir / "fig1_overall_performance.png")


def choose_representative_date(
    df: pd.DataFrame,
    model_order: list[str],
    start_hour: int,
    end_hour: int,
) -> pd.Timestamp:
    expected_hours = end_hour - start_hour + 1

    temp = df[df["hour"].between(start_hour, end_hour)].copy()
    temp["plot_date"] = temp["timestamp"].dt.normalize()

    hour_counts = (
        temp.groupby(["plot_date", "model"], observed=True)["hour"]
        .nunique()
        .unstack("model")
        .reindex(columns=model_order)
    )

    valid_dates = hour_counts[
        (hour_counts >= expected_hours).all(axis=1)
    ].index

    if len(valid_dates) == 0:
        valid_dates = hour_counts.dropna(how="all").index

    base_model = model_order[0]
    base = temp[
        (temp["model"] == base_model) &
        (temp["plot_date"].isin(valid_dates))
    ].copy()

    daily_sum = base.groupby("plot_date", observed=True)["y_true"].sum()

    if len(daily_sum) == 0:
        raise ValueError("대표 날짜를 자동 선택할 수 없어. 데이터 날짜를 확인해줘.")

    median_sum = daily_sum.median()
    chosen_date = (daily_sum - median_sum).abs().idxmin()

    return chosen_date


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


def _insert_day_gaps(plot_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    날짜가 바뀔 때 선이 전날 마지막 시간에서 다음날 첫 시간으로
    대각선 연결되지 않도록 NaN 행을 삽입한다.
    """
    plot_df = plot_df.sort_values("timestamp").copy()
    plot_df["plot_date"] = plot_df["timestamp"].dt.normalize()

    pieces = []

    for _, part in plot_df.groupby("plot_date", sort=True):
        part = part[["timestamp", value_col]].copy()
        pieces.append(part)

        # 선 끊기용 NaN
        gap = pd.DataFrame({
            "timestamp": [part["timestamp"].iloc[-1] + pd.Timedelta(seconds=1)],
            value_col: [np.nan],
        })
        pieces.append(gap)

    return pd.concat(pieces, ignore_index=True)


def _insert_day_gaps_multi(plot_df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """
    여러 값 컬럼을 가진 시계열에서도 날짜 경계에서 선/면이 이어지지 않도록
    모든 값 컬럼이 NaN인 gap 행을 삽입한다.
    """
    plot_df = plot_df.sort_values("timestamp").copy()
    plot_df["plot_date"] = plot_df["timestamp"].dt.normalize()

    pieces = []
    keep_cols = ["timestamp"] + value_cols

    for _, part in plot_df.groupby("plot_date", sort=True):
        part = part[keep_cols].copy()
        pieces.append(part)

        gap = {"timestamp": part["timestamp"].iloc[-1] + pd.Timedelta(seconds=1)}
        gap.update({col: np.nan for col in value_cols})
        pieces.append(pd.DataFrame([gap]))

    return pd.concat(pieces, ignore_index=True)


def build_all_dates_forecast_figure(
    df: pd.DataFrame,
    out_dir: Path,
    model_order: list[str],
    start_hour: int,
    end_hour: int,
) -> None:
    temp = df.copy()

    # timestamp가 datetime이 아닐 경우 대비
    temp["timestamp"] = pd.to_datetime(temp["timestamp"])

    # 지정한 발전 시간대만 사용
    temp = temp[temp["hour"].between(start_hour, end_hour)].copy()

    if temp.empty:
        raise ValueError(
            f"{start_hour}~{end_hour}시에 해당하는 데이터가 없습니다."
        )

    # 실제값은 모델별로 동일하므로 timestamp 기준 평균으로 하나만 사용
    actual = (
        temp.groupby("timestamp", observed=True)["y_true"]
        .mean()
        .reset_index()
        .sort_values("timestamp")
    )

    actual_plot = _insert_day_gaps(actual, "y_true")

    available_models = [
        model_name
        for model_name in model_order
        if not temp[temp["model"] == model_name].empty
    ]

    if not available_models:
        raise ValueError("Figure 2를 그릴 수 있는 모델 데이터가 없습니다.")

    n_models = len(available_models)
    fig_h = max(3.0 * n_models, 4.0)
    fig, axes = plt.subplots(
        n_models,
        1,
        figsize=(14, fig_h),
        sharex=True,
        squeeze=False,
    )

    axes_flat = axes.ravel()

    for ax, model_name in zip(axes_flat, available_models):
        sub = temp[temp["model"] == model_name].copy()

        pred = (
            sub.groupby("timestamp", observed=True)["q50"]
            .mean()
            .reset_index()
            .sort_values("timestamp")
        )

        pred_plot = _insert_day_gaps(pred, "q50")

        ax.plot(
            actual_plot["timestamp"],
            actual_plot["y_true"],
            linewidth=1.8,
            label="Actual",
            color="black",
            alpha=0.8,
        )

        ax.plot(
            pred_plot["timestamp"],
            pred_plot["q50"],
            linewidth=1.4,
            label="Pred q50",
        )

        ax.set_title(str(model_name), loc="left")
        ax.set_ylabel("PV output")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")

    axes_flat[-1].set_xlabel("Date")

    # 날짜 x축 자동 정리
    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    axes_flat[-1].xaxis.set_major_locator(locator)
    axes_flat[-1].xaxis.set_major_formatter(formatter)

    fig.suptitle("Forecast comparison over all dates by model", y=1.0)
    fig.tight_layout()

    save_figure(fig, out_dir / "fig2_all_dates_forecast.png")


def build_quantile_interval_forecast_figure(
    df: pd.DataFrame,
    out_dir: Path,
    model_order: list[str],
    start_hour: int,
    end_hour: int,
) -> None:
    temp = df.copy()
    temp["timestamp"] = pd.to_datetime(temp["timestamp"])
    temp = temp[temp["hour"].between(start_hour, end_hour)].copy()

    if temp.empty:
        raise ValueError(
            f"{start_hour}~{end_hour}시에 해당하는 데이터가 없습니다."
        )

    actual = (
        temp.groupby("timestamp", observed=True)["y_true"]
        .mean()
        .reset_index()
        .sort_values("timestamp")
    )
    actual_plot = _insert_day_gaps(actual, "y_true")

    available_models = [
        model_name
        for model_name in model_order
        if not temp[temp["model"] == model_name].empty
    ]

    if not available_models:
        raise ValueError("Figure 6을 그릴 수 있는 모델 데이터가 없습니다.")

    n_models = len(available_models)
    fig_h = max(3.2 * n_models, 4.5)
    fig, axes = plt.subplots(
        n_models,
        1,
        figsize=(14, fig_h),
        sharex=True,
        squeeze=False,
    )
    axes_flat = axes.ravel()

    actual_x = mdates.date2num(actual_plot["timestamp"].dt.to_pydatetime())

    for ax, model_name in zip(axes_flat, available_models):
        sub = temp[temp["model"] == model_name].copy()

        pred = (
            sub.groupby("timestamp", observed=True)
            .agg(
                q10=("q10", "mean"),
                q50=("q50", "mean"),
                q90=("q90", "mean"),
                lower=("lower", "mean"),
                upper=("upper", "mean"),
            )
            .reset_index()
            .sort_values("timestamp")
        )

        pred_plot = _insert_day_gaps_multi(
            pred,
            ["q10", "q50", "q90", "lower", "upper"],
        )
        pred_x = mdates.date2num(pred_plot["timestamp"].dt.to_pydatetime())

        ax.fill_between(
            pred_x,
            pred_plot["lower"].to_numpy(dtype=float),
            pred_plot["upper"].to_numpy(dtype=float),
            color="#3182bd",
            alpha=0.35,
            label="q10-q90 interval",
            linewidth=0,
        )

        ax.plot(
            pred_x,
            pred_plot["q10"],
            color="#08519c",
            linewidth=1.3,
            linestyle="--",
            alpha=0.95,
            label="Pred q10",
        )
        ax.plot(
            pred_x,
            pred_plot["q50"],
            color="#d94801",
            linewidth=1.7,
            alpha=0.98,
            label="Pred q50",
        )
        ax.plot(
            pred_x,
            pred_plot["q90"],
            color="#006d2c",
            linewidth=1.3,
            linestyle="--",
            alpha=0.95,
            label="Pred q90",
        )
        ax.plot(
            actual_x,
            actual_plot["y_true"],
            color="black",
            linewidth=1.4,
            alpha=0.38,
            label="Actual",
            zorder=4,
        )

        ax.set_title(str(model_name), loc="left")
        ax.set_ylabel("PV output")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", ncol=2)

    axes_flat[-1].set_xlabel("Date")

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    axes_flat[-1].xaxis.set_major_locator(locator)
    axes_flat[-1].xaxis.set_major_formatter(formatter)

    fig.suptitle("Quantile forecast intervals over all dates by model", y=1.0)
    fig.tight_layout()

    save_figure(fig, out_dir / "fig6_quantile_interval_forecast.png")


def _timestamp_prediction_table(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate overlapping forecasts to one prediction per model/timestamp."""
    pred = (
        df.groupby(["model", "timestamp"], observed=True)
        .agg(
            y_true=("y_true", "mean"),
            q10=("q10", "mean"),
            q50=("q50", "mean"),
            q90=("q90", "mean"),
            hour=("hour", "first"),
        )
        .reset_index()
    )
    pred["date"] = pred["timestamp"].dt.normalize()
    pred["lower"] = np.minimum(pred["q10"], pred["q90"])
    pred["upper"] = np.maximum(pred["q10"], pred["q90"])
    pred["abs_error"] = (pred["y_true"] - pred["q50"]).abs()
    return pred


def _resolve_selection_model(
    available_models: list[str],
    selection_model: str | None,
) -> str | None:
    """Resolve the model used for objective representative-day selection."""
    if selection_model is not None:
        exact = [m for m in available_models if str(m) == selection_model]
        if exact:
            return exact[0]
        partial = [
            m for m in available_models
            if selection_model.lower() in str(m).lower()
        ]
        if len(partial) == 1:
            return partial[0]
        raise ValueError(
            f"selection_model={selection_model!r}을 모델 목록에서 찾을 수 없어. "
            f"available={available_models}"
        )

    mamba_models = [m for m in available_models if "mamba" in str(m).lower()]
    return mamba_models[0] if len(mamba_models) == 1 else None


def _centered_date_window(
    dates: pd.DatetimeIndex,
    anchor: pd.Timestamp,
    n_days: int,
) -> pd.DatetimeIndex:
    if n_days not in {2, 3}:
        raise ValueError(f"period_days는 2 또는 3이어야 해. got={n_days}")
    if len(dates) < n_days:
        raise ValueError(
            f"완전한 날짜가 {len(dates)}일뿐이라 {n_days}일 구간을 만들 수 없어."
        )

    anchor_pos = int(dates.get_indexer([anchor])[0])
    left = (n_days - 1) // 2
    start = max(0, min(anchor_pos - left, len(dates) - n_days))
    return dates[start : start + n_days]


def select_representative_and_rapid_periods(
    df: pd.DataFrame,
    model_order: list[str],
    start_hour: int,
    end_hour: int,
    period_days: int = 3,
    selection_model: str | None = None,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, pd.DataFrame]:
    """Select objective normal and rapid-change windows.

    The representative anchor is selected from the central 50% of the test
    dates. Its daily MAE must be closest to the median daily MAE over the full
    test set. Mamba-TFT is used when its label can be identified uniquely;
    otherwise the mean daily MAE across all models is used.

    The rapid-change anchor maximizes the mean absolute hour-to-hour ramp of
    actual PV. Both anchors are expanded to a contiguous 2- or 3-day window.
    """
    temp = df[df["hour"].between(start_hour, end_hour)].copy()
    pred = _timestamp_prediction_table(temp)
    available_models = [
        model for model in model_order
        if not pred[pred["model"] == model].empty
    ]
    if not available_models:
        raise ValueError("기간을 선택할 수 있는 모델 예측 데이터가 없어.")

    expected_hours = end_hour - start_hour + 1
    counts = (
        pred.groupby(["date", "model"], observed=True)["hour"]
        .nunique()
        .unstack("model")
        .reindex(columns=available_models)
    )
    valid_dates = pd.DatetimeIndex(
        counts.index[(counts >= expected_hours).all(axis=1)]
    ).sort_values()
    if len(valid_dates) < period_days:
        raise ValueError("모든 모델에서 완전한 예측을 가진 날짜가 부족해.")

    daily_by_model = (
        pred[pred["date"].isin(valid_dates)]
        .groupby(["date", "model"], observed=True)["abs_error"]
        .mean()
        .unstack("model")
        .reindex(index=valid_dates, columns=available_models)
    )
    resolved_model = _resolve_selection_model(available_models, selection_model)
    if resolved_model is None:
        representative_error = daily_by_model.mean(axis=1)
        selection_label = "all-model mean"
    else:
        representative_error = daily_by_model[resolved_model]
        selection_label = str(resolved_model)

    median_daily_mae = float(representative_error.median())
    # "test set 중앙부"를 객관화: 완전한 날짜의 중앙 50%만 후보로 둔다.
    central_start = len(valid_dates) // 4
    central_stop = len(valid_dates) - central_start
    central_dates = valid_dates[central_start:central_stop]
    representative_anchor = (
        representative_error.loc[central_dates] - median_daily_mae
    ).abs().idxmin()
    representative_dates = _centered_date_window(
        valid_dates, representative_anchor, period_days
    )

    actual = (
        pred.groupby("timestamp", observed=True)["y_true"]
        .mean()
        .reset_index()
        .sort_values("timestamp")
    )
    actual["date"] = actual["timestamp"].dt.normalize()
    actual = actual[actual["date"].isin(valid_dates)].copy()
    actual["abs_ramp"] = actual.groupby("date", observed=True)["y_true"].diff().abs()
    ramp = actual.groupby("date", observed=True)["abs_ramp"].agg(
        mean_abs_ramp="mean",
        max_abs_ramp="max",
    )
    rapid_anchor = ramp.sort_values(
        ["mean_abs_ramp", "max_abs_ramp"], ascending=False
    ).index[0]
    rapid_dates = _centered_date_window(valid_dates, rapid_anchor, period_days)

    selection = pd.DataFrame(index=valid_dates)
    selection.index.name = "date"
    selection["selection_model"] = selection_label
    selection["daily_mae"] = representative_error.reindex(valid_dates)
    selection["median_daily_mae"] = median_daily_mae
    selection["distance_from_median_mae"] = (
        selection["daily_mae"] - median_daily_mae
    ).abs()
    selection["is_central_candidate"] = selection.index.isin(central_dates)
    selection["mean_abs_actual_ramp"] = ramp["mean_abs_ramp"].reindex(valid_dates)
    selection["max_abs_actual_ramp"] = ramp["max_abs_ramp"].reindex(valid_dates)
    selection["representative_anchor"] = selection.index == representative_anchor
    selection["representative_period"] = selection.index.isin(representative_dates)
    selection["rapid_change_anchor"] = selection.index == rapid_anchor
    selection["rapid_change_period"] = selection.index.isin(rapid_dates)
    return representative_dates, rapid_dates, selection.reset_index()


def _build_selected_period_quantile_figure(
    df: pd.DataFrame,
    out_path: Path,
    model_order: list[str],
    selected_dates: pd.DatetimeIndex,
    title: str,
) -> None:
    pred = _timestamp_prediction_table(df)
    pred = pred[pred["date"].isin(selected_dates)].copy()
    available_models = [
        model for model in model_order
        if not pred[pred["model"] == model].empty
    ]
    if not available_models:
        raise ValueError(f"{title}에 표시할 모델 데이터가 없어.")

    actual = (
        pred.groupby("timestamp", observed=True)["y_true"]
        .mean()
        .reset_index()
        .sort_values("timestamp")
    )
    actual_plot = _insert_day_gaps(actual, "y_true")
    actual_x = mdates.date2num(actual_plot["timestamp"].dt.to_pydatetime())

    fig, axes = plt.subplots(
        len(available_models),
        1,
        figsize=(14, max(3.2 * len(available_models), 4.5)),
        sharex=True,
        squeeze=False,
    )
    for ax, model_name in zip(axes.ravel(), available_models):
        model_pred = pred[pred["model"] == model_name].sort_values("timestamp")
        model_plot = _insert_day_gaps_multi(
            model_pred, ["q10", "q50", "q90", "lower", "upper"]
        )
        x = mdates.date2num(model_plot["timestamp"].dt.to_pydatetime())

        ax.fill_between(
            x,
            model_plot["lower"].to_numpy(dtype=float),
            model_plot["upper"].to_numpy(dtype=float),
            color="#3182bd",
            alpha=0.28,
            linewidth=0,
            label="q10-q90 interval",
        )
        ax.plot(x, model_plot["q10"], "--", color="#08519c", linewidth=1.2, label="q10")
        ax.plot(x, model_plot["q50"], color="#d94801", linewidth=1.8, label="q50")
        ax.plot(x, model_plot["q90"], "--", color="#006d2c", linewidth=1.2, label="q90")
        ax.plot(actual_x, actual_plot["y_true"], color="black", linewidth=1.6, label="Actual", zorder=4)
        ax.set_title(str(model_name), loc="left")
        ax.set_ylabel("PV output")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", ncol=3)

    axes.ravel()[-1].set_xlabel("Time")
    locator = mdates.AutoDateLocator()
    axes.ravel()[-1].xaxis.set_major_locator(locator)
    axes.ravel()[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    date_text = f"{selected_dates[0].date()} to {selected_dates[-1].date()}"
    fig.suptitle(f"{title}\n{date_text}", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, out_path)


def build_representative_and_rapid_change_figures(
    df: pd.DataFrame,
    out_dir: Path,
    model_order: list[str],
    start_hour: int,
    end_hour: int,
    period_days: int = 3,
    selection_model: str | None = None,
) -> None:
    representative_dates, rapid_dates, selection = (
        select_representative_and_rapid_periods(
            df=df,
            model_order=model_order,
            start_hour=start_hour,
            end_hour=end_hour,
            period_days=period_days,
            selection_model=selection_model,
        )
    )
    selection.to_csv(
        out_dir / "fig7_fig8_period_selection.csv",
        index=False,
        encoding="utf-8-sig",
    )
    _build_selected_period_quantile_figure(
        df=df,
        out_path=out_dir / "fig7_representative_period.png",
        model_order=model_order,
        selected_dates=representative_dates,
        title="Representative period: actual and quantile forecasts",
    )
    _build_selected_period_quantile_figure(
        df=df,
        out_path=out_dir / "fig8_rapid_change_period.png",
        model_order=model_order,
        selected_dates=rapid_dates,
        title="Rapid-change period: actual and quantile forecasts",
    )


def build_hourly_mae_heatmap_4models(
    df: pd.DataFrame,
    out_dir: Path,
    model_order: list[str],
    start_hour: int,
    end_hour: int,
) -> None:
    hours = list(range(start_hour, end_hour + 1))

    mae_matrix = (
        df[df["hour"].between(start_hour, end_hour)]
        .groupby(["model", "hour"], observed=True)["abs_error"]
        .mean()
        .unstack("hour")
        .reindex(index=model_order, columns=hours)
    )

    mae_matrix.to_csv(
        out_dir / "fig3_hourly_mae_heatmap.csv",
        encoding="utf-8-sig",
    )

    plot_heatmap(
        matrix=mae_matrix,
        title="Hourly MAE heatmap by model",
        xlabel="Hour of day",
        ylabel="Model",
        colorbar_label="MAE",
        out_path=out_dir / "fig3_hourly_mae_heatmap.png",
        value_format=".1f",
    )


def build_coverage_width_tradeoff_figure(
    summary: pd.DataFrame,
    out_dir: Path,
    nominal_coverage: float = 0.8,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))

    x = summary["width_10_90"].to_numpy()
    y = summary["coverage_10_90"].to_numpy()
    models = summary["model"].astype(str).tolist()

    ax.scatter(x, y, s=90)

    for xi, yi, model_name in zip(x, y, models):
        ax.annotate(
            model_name,
            xy=(xi, yi),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
        )

    ax.axhline(
        nominal_coverage,
        linestyle="--",
        linewidth=1.2,
        label=f"Nominal coverage = {nominal_coverage:.1f}",
    )

    ax.set_title("Coverage-width trade-off")
    ax.set_xlabel("Mean prediction interval width q10-q90")
    ax.set_ylabel("Empirical coverage q10-q90")
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    save_figure(fig, out_dir / "fig4_coverage_width_tradeoff.png")


def build_quantile_risk_figure(
    summary: pd.DataFrame,
    out_dir: Path,
) -> None:
    models = summary["model"].astype(str).tolist()
    quantiles = ["q10", "q50", "q90"]

    x = np.arange(len(quantiles))
    n_models = len(models)
    bar_width = 0.8 / max(n_models, 1)

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, model_name in enumerate(models):
        row = summary[summary["model"].astype(str) == model_name].iloc[0]

        values = [
            row["qrisk_q10"],
            row["qrisk_q50"],
            row["qrisk_q90"],
        ]

        offset = (i - (n_models - 1) / 2) * bar_width

        ax.bar(
            x + offset,
            values,
            width=bar_width,
            label=model_name,
        )

    ax.set_title("Quantile risk comparison")
    ax.set_xlabel("Quantile")
    ax.set_ylabel("QRisk")
    ax.set_xticks(x)
    ax.set_xticklabels(quantiles)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=2)

    fig.tight_layout()
    save_figure(fig, out_dir / "fig5_quantile_risk_comparison.png")

def summarize(
    df: pd.DataFrame,
    out_dir: Path,
    model_order: list[str],
) -> pd.DataFrame:
    summary = compute_model_metrics(df, model_order=model_order)

    summary.to_csv(
        out_dir / "model_metric_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print(summary)

    return summary

def plot_heatmaps_main(
    pred_csvs: list[str],
    labels: list[str],
    out_dir: str | Path,
    horizon_csvs: list[str] | None = None,
    start_hour: int = 7,
    end_hour: int = 17,
    period_days: int = 3,
    selection_model: str | None = None,
) -> None:
    if len(pred_csvs) != len(labels):
        raise ValueError(
            f"--pred_csv 개수와 --label 개수가 같아야 해. "
            f"pred_csv={len(pred_csvs)}, label={len(labels)}"
        )
    if horizon_csvs is not None and len(horizon_csvs) != len(labels):
        raise ValueError(
            f"--horizon_csv 개수와 --label 개수가 같아야 해. "
            f"horizon_csv={len(horizon_csvs)}, label={len(labels)}"
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = []
    for path, label in zip(pred_csvs, labels):
        frames.append(load_prediction_csv(path, label))

    df = pd.concat(frames, ignore_index=True)

    # 모델 출력 순서 고정
    df["model"] = pd.Categorical(
        df["model"],
        categories=labels,
        ordered=True,
    )

    df = df.sort_values(["model", "timestamp"]).reset_index(drop=True)

    # 원본 집계용 long-format 데이터 저장
    df.to_csv(
        out_dir / "figure_input_long.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 전체 지표 요약
    summary = summarize(df, out_dir, model_order=labels)

    # Fig. 1 전체 성능 비교
    build_overall_performance_figure(
        summary=summary,
        out_dir=out_dir,
    )

    # Fig. 2 전체 날짜 예측 시계열 비교
    build_all_dates_forecast_figure(
        df=df,
        out_dir=out_dir,
        model_order=labels,
        start_hour=start_hour,
        end_hour=end_hour,
    )

    # Fig. 3 시간대별 MAE heatmap
    build_hourly_mae_heatmap_4models(
        df=df,
        out_dir=out_dir,
        model_order=labels,
        start_hour=start_hour,
        end_hour=end_hour,
    )

    # Fig. 4 Coverage-width trade-off
    build_coverage_width_tradeoff_figure(
        summary=summary,
        out_dir=out_dir,
        nominal_coverage=0.8,
    )

    # Fig. 5 Quantile risk 비교
    build_quantile_risk_figure(
        summary=summary,
        out_dir=out_dir,
    )

    # Fig. 6 모델별 quantile 예측 구간 비교
    build_quantile_interval_forecast_figure(
        df=df,
        out_dir=out_dir,
        model_order=labels,
        start_hour=start_hour,
        end_hour=end_hour,
    )

    # Fig. 7~8 객관적으로 선택한 대표/급변 구간의 quantile 예측 비교
    build_representative_and_rapid_change_figures(
        df=df,
        out_dir=out_dir,
        model_order=labels,
        start_hour=start_hour,
        end_hour=end_hour,
        period_days=period_days,
        selection_model=selection_model,
    )

    # Fig. 9 forecast origin별 전체 출력을 보존한 horizon-wise q50 MAE
    if horizon_csvs is not None:
        horizon_frames = [
            load_horizon_prediction_csv(path, label)
            for path, label in zip(horizon_csvs, labels)
        ]
        horizon_df = pd.concat(horizon_frames, ignore_index=True)
        build_horizon_wise_mae_figure(
            horizon_df=horizon_df,
            out_dir=out_dir,
            model_order=labels,
        )
        tft_models = [
            label for label in labels
            if str(label).lower() in {"tft", "mamba-tft", "mamba_tft"}
        ]
        if len(tft_models) == 2:
            build_horizon_wise_mae_figure(
                horizon_df=horizon_df[horizon_df["model"].isin(tft_models)].copy(),
                out_dir=out_dir,
                model_order=tft_models,
                file_stem="fig9b_tft_mamba_horizon_wise_mae",
                title="Horizon-wise MAE: TFT vs Mamba-TFT",
            )

    # 기존 heatmap도 같이 저장하고 싶으면 유지
    build_hourly_coverage_heatmap(
        df=df,
        out_dir=out_dir,
        start_hour=start_hour,
        end_hour=end_hour,
        nominal_coverage=0.8,
    )

    build_month_hour_mae_heatmap(
        df=df,
        out_dir=out_dir,
        start_hour=start_hour,
        end_hour=end_hour,
    )

    print(f"\nSaved figures to: {out_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pred_csv",
        action="append",
        required=True,
        help="prediction csv path. 여러 모델 비교 시 반복해서 입력.",
    )
    parser.add_argument(
        "--label",
        action="append",
        required=True,
        help="각 prediction csv에 대응되는 모델 이름. --pred_csv와 같은 개수.",
    )
    parser.add_argument(
        "--horizon_csv",
        action="append",
        help=(
            "forecast origin과 horizon이 모두 보존된 prediction CSV. "
            "Fig. 9 생성 시 모델별로 반복해서 입력."
        ),
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="그림과 집계 csv를 저장할 폴더.",
    )
    parser.add_argument("--start_hour", type=int, default=7)
    parser.add_argument("--end_hour", type=int, default=17)
    parser.add_argument(
        "--period_days",
        type=int,
        choices=[2, 3],
        default=3,
        help="대표/급변 구간 길이(일).",
    )
    parser.add_argument(
        "--selection_model",
        help=(
            "대표 날짜의 daily MAE를 계산할 모델 label. 생략하면 label에 "
            "'mamba'가 하나만 있을 때 해당 모델, 아니면 전체 모델 평균을 사용."
        ),
    )

    args = parser.parse_args()

    plot_heatmaps_main(
        pred_csvs=args.pred_csv,
        labels=args.label,
        out_dir=args.out_dir,
        horizon_csvs=args.horizon_csv,
        start_hour=args.start_hour,
        end_hour=args.end_hour,
        period_days=args.period_days,
        selection_model=args.selection_model,
    )

if __name__ == "__main__":
    main()
