from __future__ import annotations

from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_forecasting.metrics import QuantileLoss

try:
    # Keep the standalone baseline on the exact implementation used by Mamba-TFT.
    from mamba_ssm import Mamba
except ImportError:  # pragma: no cover - exercised only in incomplete environments
    Mamba = None


class SimpleMambaForecastingModel(nn.Module):
    """Past-only Mamba backbone with a direct multi-horizon quantile head."""

    def __init__(
        self,
        input_size: int,
        prediction_length: int,
        quantiles: list[float],
        d_model: int = 64,
        num_mamba_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        use_future_known_inputs: bool = False,
    ) -> None:
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "SimpleMambaForecastingModel requires mamba-ssm, which is also "
                "required by the repository's Mamba-TFT implementation."
            )
        if prediction_length < 1 or not quantiles:
            raise ValueError("prediction_length and quantiles must be non-empty")
        if num_mamba_layers < 1:
            raise ValueError("num_mamba_layers must be at least 1")

        self.input_size = input_size
        self.prediction_length = prediction_length
        self.quantiles = list(quantiles)
        self.d_model = d_model
        self.use_future_known_inputs = use_future_known_inputs

        self.input_projection = nn.Linear(input_size, d_model)
        self.mamba_blocks = nn.ModuleList(
            [
                Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(num_mamba_layers)
            ]
        )
        # mamba_ssm.Mamba itself does not add an outer residual/norm. Match the
        # repository's Mamba-TFT temporal stack: Mamba -> dropout -> residual -> norm.
        self.block_dropout = nn.Dropout(dropout)
        self.block_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(num_mamba_layers)]
        )
        self.final_norm = nn.LayerNorm(d_model)
        output_size = (
            len(self.quantiles)
            if use_future_known_inputs
            else prediction_length * len(self.quantiles)
        )
        self.output_layer = nn.Linear(d_model, output_size)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected input [B,L,F], got {tuple(x.shape)}")
        if x.size(-1) != self.input_size:
            raise ValueError(
                f"Expected input_size={self.input_size}, got F={x.size(-1)}"
            )
        hidden = self.input_projection(x)
        for block, norm in zip(self.mamba_blocks, self.block_norms):
            residual = hidden
            hidden = norm(residual + self.block_dropout(block(hidden)))
        return self.final_norm(hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.encode(x)
        if self.use_future_known_inputs:
            if hidden.size(1) < self.prediction_length:
                raise ValueError("Input sequence is shorter than prediction_length")
            # One causal representation per future-known position. Horizon h
            # cannot see covariates from horizons after h.
            decoder_hidden = hidden[:, -self.prediction_length :, :]
            return self.output_layer(decoder_hidden)

        history_context = hidden[:, -1, :]
        return self.output_layer(history_context).view(
            x.size(0), self.prediction_length, len(self.quantiles)
        )


class SimpleMambaLightningModule(pl.LightningModule):
    def __init__(
        self,
        input_size: int,
        prediction_length: int,
        quantiles: list[float] | None = None,
        d_model: int = 64,
        num_mamba_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        use_future_known_inputs: bool = False,
    ) -> None:
        super().__init__()
        quantiles = quantiles or [0.1, 0.5, 0.9]
        self.save_hyperparameters()
        self.quantiles = list(quantiles)
        self.quantile_loss = QuantileLoss(self.quantiles)
        self.model = SimpleMambaForecastingModel(
            input_size=input_size,
            prediction_length=prediction_length,
            quantiles=self.quantiles,
            d_model=d_model,
            num_mamba_layers=num_mamba_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout,
            use_future_known_inputs=use_future_known_inputs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        x, y, _ = batch
        y_hat = self(x)
        loss = self.quantile_loss.loss(y_hat, y).mean()
        p50 = y_hat[:, :, self.quantiles.index(0.5)]
        mae = F.l1_loss(p50, y)
        rmse = torch.sqrt(F.mse_loss(p50, y))
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_mae", mae, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_rmse", rmse, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0) -> dict[str, Any]:
        x, y, meta = batch
        return {
            "y_hat": self(x).detach().cpu(),
            "y": y.detach().cpu(),
            "meta": meta,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }


def parameter_counts(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable
