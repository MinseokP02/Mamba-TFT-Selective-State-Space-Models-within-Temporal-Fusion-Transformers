# from pytorch_forecasting import TemporalFusionTransformer
from pv_forecast.models.tft_vendor import TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss

# def build_tft(training_ds, cfg: dict):
#     m = cfg["tft"]
#     return TemporalFusionTransformer.from_dataset(
#         training_ds,
#         hidden_size=m["hidden_size"],
#         lstm_layers=m["lstm_layers"],
#         dropout=m["dropout"],
#         attention_head_size=m["attention_head_size"],
#         learning_rate=m["learning_rate"],
#         loss=QuantileLoss(m["quantiles"]),
#     )

def build_tft(training_ds, cfg: dict):
    m = cfg["tft"]
    return TemporalFusionTransformer.from_dataset(
        training_ds,
        hidden_size=m["hidden_size"],
        lstm_layers=m["lstm_layers"],
        dropout=m["dropout"],
        attention_head_size=m["attention_head_size"],
        learning_rate=m["learning_rate"],
        loss=QuantileLoss(m["quantiles"]),
    )