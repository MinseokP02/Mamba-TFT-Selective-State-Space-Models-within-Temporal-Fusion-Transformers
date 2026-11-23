from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from scripts.run_all_cnn2 import CNN2QuantileLightningModule
from scripts.run_all_cnn2 import main as run_all_cnn2_main


class PaperCNN1DQuantileNet(nn.Module):
    """
    Paper-style compact CNN adapted to causal multi-horizon quantile forecasting.

    Paper:
      Conv1D(64, k=3) -> GlobalMaxPool -> Dense(64) -> Dense(8) -> Dense(1)

    This implementation:
      causal Conv1D -> cumulative max pool -> per-horizon Dense(64)
      -> Dense(8) -> Dense(num_quantiles)

    The input contains encoder history followed by decoder rows containing only
    future-known covariates.  Output horizon h is computed from the cumulative
    representation at decoder position h, so it cannot use decoder covariates
    from positions after h.
    """

    def __init__(
        self,
        num_features: int,
        sequence_length: int,
        output_dim: int,
        quantiles: list[float],
        conv_out_channels: int = 64,
        kernel_size: int = 3,
        hidden_dim1: int = 64,
        hidden_dim2: int = 8,
        dropout: float = 0.0,
        activation: str = "relu",
    ):
        super().__init__()
        if sequence_length < kernel_size:
            raise ValueError(
                f"sequence_length({sequence_length}) must be >= "
                f"kernel_size({kernel_size})"
            )

        act = nn.GELU if activation == "gelu" else nn.ReLU
        self.output_dim = output_dim
        self.num_quantiles = len(quantiles)
        self.kernel_size = kernel_size

        self.conv = nn.Conv1d(
            num_features,
            conv_out_channels,
            kernel_size,
            padding=0,
        )
        self.activation = act()
        self.head = nn.Sequential(
            nn.Linear(conv_out_channels, hidden_dim1),
            act(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim1, hidden_dim2),
            act(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_dim2, self.num_quantiles),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(-1) < self.output_dim:
            raise ValueError(
                f"sequence length ({x.size(-1)}) must be at least "
                f"output_dim ({self.output_dim})"
            )

        # Left-only padding makes the convolution causal while preserving the
        # input sequence length. Output position t can only depend on inputs
        # at t and earlier positions.
        x = F.pad(x, (self.kernel_size - 1, 0))
        x = self.activation(self.conv(x))

        # A global max pool would mix all decoder positions. Cumulative max
        # pooling preserves the paper-style temporal aggregation but restricts
        # every position to the prefix available at that position.
        x = torch.cummax(x, dim=-1).values

        # The final output_dim positions correspond to decoder horizons. The
        # shared head maps each causal decoder representation independently to
        # its quantiles: (B, C, H) -> (B, H, Q).
        decoder_x = x[:, :, -self.output_dim :].transpose(1, 2)
        return self.head(decoder_x)


class PaperCNN1DQuantileModule(CNN2QuantileLightningModule):
    """Reuse CNN2's quantile loss, metrics, optimizer, and prediction flow."""

    def __init__(
        self,
        num_features: int,
        sequence_length: int,
        output_dim: int,
        quantiles: list[float],
        conv_out_channels: int = 64,
        kernel_size: int = 3,
        hidden_dim1: int = 64,
        hidden_dim2: int = 8,
        dropout: float = 0.0,
        activation: str = "relu",
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        optimizer_name: str = "adamw",
    ):
        super().__init__(
            num_features=num_features,
            sequence_length=sequence_length,
            output_dim=output_dim,
            quantiles=quantiles,
            conv_out_channels=conv_out_channels,
            kernel_size=kernel_size,
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2,
            dropout=dropout,
            activation=activation,
            lr=lr,
            weight_decay=weight_decay,
            optimizer_name=optimizer_name,
        )
        self.model = PaperCNN1DQuantileNet(
            num_features=num_features,
            sequence_length=sequence_length,
            output_dim=output_dim,
            quantiles=quantiles,
            conv_out_channels=conv_out_channels,
            kernel_size=kernel_size,
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2,
            dropout=dropout,
            activation=activation,
        )


def main(config_path: str) -> None:
    run_all_cnn2_main(
        config_path,
        model_module_cls=PaperCNN1DQuantileModule,
        run_suffix="_cnn3_all",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
