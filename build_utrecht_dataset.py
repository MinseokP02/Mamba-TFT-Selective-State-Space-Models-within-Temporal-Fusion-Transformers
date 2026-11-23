import argparse
import json
from pathlib import Path
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests


UTRECHT_LAT = 52.0907
UTRECHT_LON = 5.1214

OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

WEATHER_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "cloud_cover",
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
    "sunshine_duration",
    "wind_speed_10m",
]


def detect_timestamp_col(columns):
    candidates = [
        "timestamp",
        "time",
        "datetime",
        "date",
        "DateTime",
        "Timestamp",
        "Unnamed: 0",
    ]

    for c in candidates:
        if c in columns:
            return c

    return columns[0]


def detect_long_format(columns, id_col, value_col):
    if id_col is not None and value_col is not None:
        return True

    possible_id_cols = ["id", "ID", "system_id", "site_id", "pv_id"]
    possible_value_cols = [
        "power",
        "Power",
        "pv_power",
        "pv_power_w",
        "power_w",
        "measurement",
        "value",
    ]

    has_id = any(c in columns for c in possible_id_cols)
    has_value = any(c in columns for c in possible_value_cols)

    return has_id and has_value


def pick_existing_col(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def read_single_pv_id(
    pv_csv,
    system_id=None,
    timestamp_col=None,
    id_col=None,
    value_col=None,
    pv_format="auto",
):
    pv_csv = Path(pv_csv)

    if not pv_csv.exists():
        raise FileNotFoundError("PV CSV file not found: {}".format(pv_csv))

    columns = list(pd.read_csv(pv_csv, nrows=0).columns)

    if timestamp_col is None:
        timestamp_col = detect_timestamp_col(columns)

    if pv_format == "auto":
        is_long = detect_long_format(columns, id_col, value_col)
    elif pv_format == "long":
        is_long = True
    elif pv_format == "wide":
        is_long = False
    else:
        raise ValueError("pv_format must be one of: auto, wide, long")

    if is_long:
        if id_col is None:
            id_col = pick_existing_col(columns, ["id", "ID", "system_id", "site_id", "pv_id"])
        if value_col is None:
            value_col = pick_existing_col(
                columns,
                [
                    "power",
                    "Power",
                    "pv_power",
                    "pv_power_w",
                    "power_w",
                    "measurement",
                    "value",
                ],
            )

        if id_col is None or value_col is None:
            raise ValueError(
                "Long format detected, but id_col/value_col could not be detected. "
                "Use --id-col and --value-col."
            )

        usecols = [timestamp_col, id_col, value_col]

        selected_chunks = []
        first_id = None

        for chunk in pd.read_csv(pv_csv, usecols=usecols, chunksize=1_000_000):
            if first_id is None:
                non_null_ids = chunk[id_col].dropna().astype(str).unique()
                if len(non_null_ids) == 0:
                    continue
                first_id = non_null_ids[0]

            target_id = str(system_id) if system_id is not None else str(first_id)
            mask = chunk[id_col].astype(str) == target_id
            selected = chunk.loc[mask, [timestamp_col, value_col]].copy()

            if len(selected) > 0:
                selected_chunks.append(selected)

        if not selected_chunks:
            raise ValueError("No rows found for system_id={}".format(system_id))

        df = pd.concat(selected_chunks, ignore_index=True)
        selected_id = str(system_id) if system_id is not None else str(first_id)

        df = df.rename(
            columns={
                timestamp_col: "timestamp",
                value_col: "pv_output",
            }
        )

    else:
        id_columns = [c for c in columns if c != timestamp_col]

        if len(id_columns) == 0:
            raise ValueError("No PV ID columns found in wide CSV.")

        if system_id is None:
            selected_id = str(id_columns[0])
        else:
            selected_id = str(system_id)

        if selected_id not in columns:
            raise ValueError(
                "system_id '{}' not found as a column. Available examples: {}".format(
                    selected_id,
                    id_columns[:10],
                )
            )

        df = pd.read_csv(pv_csv, usecols=[timestamp_col, selected_id])
        df = df.rename(
            columns={
                timestamp_col: "timestamp",
                selected_id: "pv_output",
            }
        )

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["pv_output"] = pd.to_numeric(df["pv_output"], errors="coerce")

    df = df.dropna(subset=["timestamp"])
    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp"], keep="last")

    return df, selected_id


def resample_pv_to_hourly(pv_df, start_date, end_date, fill_missing_pv=0.0):
    start_ts = pd.Timestamp(start_date, tz="UTC")
    end_ts = pd.Timestamp(end_date, tz="UTC") + pd.Timedelta(hours=23)

    pv_df = pv_df.set_index("timestamp").sort_index()

    # 1분 단위 발전량을 1시간 평균 발전량으로 변환
    hourly = pv_df[["pv_output"]].resample("1H").mean()

    # 2014-01-01 00:00 ~ 2017-12-31 23:00 형태의 시간축 생성
    full_index = pd.date_range(start=start_ts, end=end_ts, freq="1H", tz="UTC")
    hourly = hourly.reindex(full_index)
    hourly.index.name = "timestamp"

    hourly["pv_missing_flag"] = hourly["pv_output"].isna().astype(int)
    hourly["pv_output"] = hourly["pv_output"].fillna(fill_missing_pv)

    hourly = hourly.reset_index()

    return hourly


def fetch_open_meteo_weather(
    latitude,
    longitude,
    start_date,
    end_date,
    model="era5",
):
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": ",".join(WEATHER_VARS),
        "timezone": "GMT",
        "models": model,
    }

    request_url = OPEN_METEO_ARCHIVE_URL + "?" + urlencode(params)
    print("[Open-Meteo request]")
    print(request_url)

    response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=120)

    if response.status_code != 200:
        raise RuntimeError(
            "Open-Meteo request failed. status_code={}, body={}".format(
                response.status_code,
                response.text[:1000],
            )
        )

    payload = response.json()

    if payload.get("error"):
        raise RuntimeError("Open-Meteo API error: {}".format(payload))

    if "hourly" not in payload:
        raise RuntimeError("No hourly weather data in response: {}".format(payload))

    weather = pd.DataFrame(payload["hourly"])

    if "time" not in weather.columns:
        raise RuntimeError("No time column in Open-Meteo hourly response.")

    weather = weather.rename(columns={"time": "timestamp"})
    weather["timestamp"] = pd.to_datetime(weather["timestamp"], utc=True, errors="coerce")

    for col in WEATHER_VARS:
        if col in weather.columns:
            weather[col] = pd.to_numeric(weather[col], errors="coerce")

    weather = weather.dropna(subset=["timestamp"])
    weather = weather.sort_values("timestamp")
    weather = weather.drop_duplicates(subset=["timestamp"], keep="last")

    return weather, request_url


def build_tft_dataset(
    pv_hourly,
    weather,
    selected_id,
    daytime_only=True,
    day_start=7,
    day_end=17,
):
    df = pd.merge(pv_hourly, weather, on="timestamp", how="left")

    # API 일부 결측 대응: 기상값은 0으로 채우기보다 시간 보간이 자연스러움
    for col in WEATHER_VARS:
        if col in df.columns:
            df[col] = df[col].interpolate(limit_direction="both")
            df[col] = df[col].ffill().bfill()

    df["series_id"] = str(selected_id)

    df["hour"] = df["timestamp"].dt.hour.astype(int)
    df["dayofweek"] = df["timestamp"].dt.dayofweek.astype(int)
    df["month"] = df["timestamp"].dt.month.astype(int)
    df["dayofyear"] = df["timestamp"].dt.dayofyear.astype(int)
    df["year"] = df["timestamp"].dt.year.astype(int)

    # 태양광 발전이 가능한 시간대 여부
    if "shortwave_radiation" in df.columns:
        df["is_daylight"] = (df["shortwave_radiation"] > 0).astype(int)
    else:
        df["is_daylight"] = 0

    # 기존 실험 흐름에 맞춰 07~17시만 사용
    if daytime_only:
        df = df[df["hour"].between(day_start, day_end)].copy()

    # PyTorch Forecasting TimeSeriesDataSet에서 쓰기 편하게 연속 time_idx 생성
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["time_idx"] = np.arange(len(df), dtype=np.int64)

    # target 이름을 명확히 유지
    df["pv_output"] = pd.to_numeric(df["pv_output"], errors="coerce").fillna(0.0)

    column_order = [
        "timestamp",
        "series_id",
        "time_idx",
        "pv_output",
        "pv_missing_flag",
        "hour",
        "dayofweek",
        "month",
        "dayofyear",
        "year",
        "is_daylight",
    ]

    for col in WEATHER_VARS:
        if col in df.columns:
            column_order.append(col)

    df = df[column_order]

    return df


def save_eda_reports(df, out_dir):
    out_dir = Path(out_dir)
    tables_dir = out_dir / "tables"
    meta_dir = out_dir / "meta"

    tables_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    numeric_summary = df.describe(include="all").transpose()
    numeric_summary.to_csv(tables_dir / "summary.csv")

    missing_rate = df.isna().mean().reset_index()
    missing_rate.columns = ["column", "missing_rate"]
    missing_rate.to_csv(tables_dir / "missing_rate.csv", index=False)

    hourly_coverage = (
        df.groupby("hour")
        .agg(
            n=("pv_output", "size"),
            pv_mean=("pv_output", "mean"),
            pv_max=("pv_output", "max"),
            radiation_mean=("shortwave_radiation", "mean"),
            missing_pv_rate=("pv_missing_flag", "mean"),
        )
        .reset_index()
    )
    hourly_coverage.to_csv(tables_dir / "hourly_summary.csv", index=False)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pv-csv",
        type=str,
        default="datasets/filtered_pv_power_measurements_ac.csv",
        help="PV measurement CSV path",
    )
    parser.add_argument(
        "--system-id",
        type=str,
        default=None,
        help="PV system ID. If omitted, first ID column is selected.",
    )
    parser.add_argument(
        "--pv-format",
        type=str,
        default="auto",
        choices=["auto", "wide", "long"],
        help="PV CSV format. Usually this dataset is wide.",
    )
    parser.add_argument("--timestamp-col", type=str, default=None)
    parser.add_argument("--id-col", type=str, default=None)
    parser.add_argument("--value-col", type=str, default=None)

    parser.add_argument("--start-date", type=str, default="2014-01-01")
    parser.add_argument("--end-date", type=str, default="2017-12-31")

    parser.add_argument("--latitude", type=float, default=UTRECHT_LAT)
    parser.add_argument("--longitude", type=float, default=UTRECHT_LON)
    parser.add_argument(
        "--weather-model",
        type=str,
        default="era5",
        help="Open-Meteo model name, e.g. era5",
    )

    parser.add_argument(
        "--daytime-only",
        action="store_true",
        help="Use only daytime range, default 07:00~17:00 if enabled.",
    )
    parser.add_argument("--day-start", type=int, default=7)
    parser.add_argument("--day-end", type=int, default=17)

    parser.add_argument(
        "--out-csv",
        type=str,
        default=None,
        help="Output TFT-ready CSV path.",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default="reports/utrecht_single_id_eda",
        help="EDA report output directory.",
    )

    args = parser.parse_args()

    pv_df, selected_id = read_single_pv_id(
        pv_csv=args.pv_csv,
        system_id=args.system_id,
        timestamp_col=args.timestamp_col,
        id_col=args.id_col,
        value_col=args.value_col,
        pv_format=args.pv_format,
    )

    print("[Selected PV ID]")
    print(selected_id)

    print("[PV raw range]")
    print(pv_df["timestamp"].min(), "~", pv_df["timestamp"].max())

    pv_hourly = resample_pv_to_hourly(
        pv_df=pv_df,
        start_date=args.start_date,
        end_date=args.end_date,
        fill_missing_pv=0.0,
    )

    weather_df, request_url = fetch_open_meteo_weather(
        latitude=args.latitude,
        longitude=args.longitude,
        start_date=args.start_date,
        end_date=args.end_date,
        model=args.weather_model,
    )

    tft_df = build_tft_dataset(
        pv_hourly=pv_hourly,
        weather=weather_df,
        selected_id=selected_id,
        daytime_only=args.daytime_only,
        day_start=args.day_start,
        day_end=args.day_end,
    )

    if args.out_csv is None:
        suffix = "daytime_{}_{:02d}_{:02d}".format(
            selected_id,
            args.day_start,
            args.day_end,
        ) if args.daytime_only else "all_hours_{}".format(selected_id)

        out_csv = Path("datasets") / "tft_utrecht_{}_weather.csv".format(suffix)
    else:
        out_csv = Path(args.out_csv)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    tft_df.to_csv(out_csv, index=False)

    save_eda_reports(tft_df, args.report_dir)

    meta = {
        "selected_id": selected_id,
        "pv_csv": str(args.pv_csv),
        "output_csv": str(out_csv),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "latitude": args.latitude,
        "longitude": args.longitude,
        "weather_model": args.weather_model,
        "weather_vars": WEATHER_VARS,
        "open_meteo_request_url": request_url,
        "daytime_only": args.daytime_only,
        "day_start": args.day_start,
        "day_end": args.day_end,
        "n_rows": int(len(tft_df)),
        "timestamp_min": str(tft_df["timestamp"].min()),
        "timestamp_max": str(tft_df["timestamp"].max()),
    }

    report_meta_dir = Path(args.report_dir) / "meta"
    report_meta_dir.mkdir(parents=True, exist_ok=True)

    with open(report_meta_dir / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("[Saved TFT dataset]")
    print(out_csv)

    print("[Saved EDA reports]")
    print(args.report_dir)

    print("[Dataset shape]")
    print(tft_df.shape)

    print("[Columns]")
    print(list(tft_df.columns))

    print("[Head]")
    print(tft_df.head())


if __name__ == "__main__":
    main()