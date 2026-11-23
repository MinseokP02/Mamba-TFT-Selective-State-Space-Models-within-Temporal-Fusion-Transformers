from pytorch_forecasting import TimeSeriesDataSet

def build_training_dataset(df_train, cfg: dict) -> TimeSeriesDataSet:
    feats = cfg["features"]
    tft = cfg["tft"]

    return TimeSeriesDataSet(
        df_train,
        time_idx="time_idx",
        target=feats["target"],
        group_ids=feats["group_ids"],
        max_encoder_length=tft["max_encoder_length"],
        max_prediction_length=tft["max_prediction_length"],
        time_varying_known_reals=feats["known_reals"],
        time_varying_unknown_reals=feats["unknown_reals"],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

def build_validation_dataset(training_ds: TimeSeriesDataSet, df_full, cutoff_time_idx: int) -> TimeSeriesDataSet:
    # ✅ 시간 분할 유지(누수 방지): cutoff부터 예측 시작
    return TimeSeriesDataSet.from_dataset(
        training_ds,
        df_full,
        min_prediction_idx=cutoff_time_idx,
        stop_randomization=True,
    )

def build_eval_dataset(
    training_ds: TimeSeriesDataSet,
    df_full,
    min_prediction_idx: int,
    max_prediction_idx: int | None = None,
) -> TimeSeriesDataSet:
    kwargs = dict(
        dataset=training_ds,
        data=df_full,
        min_prediction_idx=min_prediction_idx,
        stop_randomization=True,
    )
    if max_prediction_idx is not None:
        kwargs["max_prediction_idx"] = max_prediction_idx

    return TimeSeriesDataSet.from_dataset(**kwargs)