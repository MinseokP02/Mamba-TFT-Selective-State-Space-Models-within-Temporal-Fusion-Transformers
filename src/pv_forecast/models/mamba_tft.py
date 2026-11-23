from pytorch_forecasting.metrics import QuantileLoss

from .tft_vendor import (
    MambaTemporalFusionTransformer,
    SeparateMambaTemporalFusionTransformer,
)


def build_mamba_tft(training_ds, cfg: dict):
    m = cfg["tft"]
    architecture = m.get("mamba_architecture", "joint")
    model_class = {
        "joint": MambaTemporalFusionTransformer,
        "separate_encoder_decoder": SeparateMambaTemporalFusionTransformer,
    }.get(architecture)
    if model_class is None:
        raise ValueError(
            f"Unknown tft.mamba_architecture={architecture!r}; expected "
            "'joint' or 'separate_encoder_decoder'"
        )

    separate_kwargs = {}
    if architecture == "separate_encoder_decoder":
        for name in (
            "encoder_mamba_layers", "encoder_mamba_d_state",
            "encoder_mamba_d_conv", "encoder_mamba_expand",
            "decoder_mamba_layers", "decoder_mamba_d_state",
            "decoder_mamba_d_conv", "decoder_mamba_expand",
        ):
            if name in m:
                separate_kwargs[name] = m[name]

    return model_class.from_dataset(
        training_ds,
        hidden_size=m["hidden_size"],
        dropout=m["dropout"],
        attention_head_size=m["attention_head_size"],
        # When false, every decoder horizon can attend to representations from
        # the complete future-known window. Unknown future targets are still
        # excluded by TimeSeriesDataSet, so this does not expose future PV.
        causal_attention=m.get("causal_attention", True),
        learning_rate=m["learning_rate"],
        loss=QuantileLoss(m["quantiles"]),

        # Mamba parameters
        mamba_layers=m.get("mamba_layers", 1),
        mamba_d_state=m.get("mamba_d_state", 16),
        mamba_d_conv=m.get("mamba_d_conv", 4),
        mamba_expand=m.get("mamba_expand", 1),
        **separate_kwargs,
    )
