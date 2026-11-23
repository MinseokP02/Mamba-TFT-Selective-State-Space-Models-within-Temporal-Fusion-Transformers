import pandas as pd
import numpy as np

def build_df_full(cfg_data: dict) -> pd.DataFrame:
    csv_path = cfg_data["csv_path"]
    index_is_time = cfg_data.get("index_is_time", True)
    fillna_value = float(cfg_data.get("fillna_value", 0.0))
    real_cols = cfg_data["real_cols"]
    missing_flag_cols = cfg_data["missing_flag_cols"]
    site = cfg_data.get("group_id_value", "site0")

    start_hour = int(cfg_data["daily_grid"]["start_hour"])
    end_hour = int(cfg_data["daily_grid"]["end_hour"])

    # 1) load
    if index_is_time:
        df = pd.read_csv(csv_path, index_col=0, parse_dates=True).copy()
        df["timestamp"] = pd.to_datetime(df.index)
    else:
        df = pd.read_csv(csv_path).copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values("timestamp").reset_index(drop=True)

    # 2) base cols
    df["series_id"] = site
    df["time_idx"] = np.arange(len(df), dtype=np.int64)
    df["date"] = df["timestamp"].dt.floor("D")

    # 3) date-hour grid (7~17)
    dates = pd.date_range(start=df["date"].min(), end=df["date"].max(), freq="D")
    hours = np.arange(start_hour, end_hour + 1)

    grid = pd.MultiIndex.from_product([dates, hours], names=["date", "hour"]).to_frame(index=False)
    grid["timestamp"] = grid["date"] + pd.to_timedelta(grid["hour"], unit="h")

    # 4) merge
    df_full = grid.merge(df, on="timestamp", how="left")
    df_full["time_idx"] = np.arange(len(df_full), dtype=np.int64)
    df_full["series_id"] = site

    # time features
    df_full["hour"] = df_full["timestamp"].dt.hour
    df_full["dayofweek"] = df_full["timestamp"].dt.dayofweek
    df_full["month"] = df_full["timestamp"].dt.month
    df_full["date"] = df_full["timestamp"].dt.floor("D")

    # numeric cast
    for c in real_cols:
        if c in df_full.columns:
            df_full[c] = pd.to_numeric(df_full[c], errors="coerce")

    # missing flags BEFORE fillna
    for c in missing_flag_cols:
        if c in df_full.columns:
            df_full[f"{c}_missing"] = df_full[c].isna().astype(int)

    # clean/fill
    exist_real = [c for c in real_cols if c in df_full.columns]
    df_full[exist_real] = df_full[exist_real].replace([np.inf, -np.inf], np.nan).fillna(fillna_value)

    return df_full