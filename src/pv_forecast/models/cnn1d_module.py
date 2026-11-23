# src/pv_forecast/models/cnn1d_module.py

from __future__ import annotations

from typing import Any, Dict

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1DForecaster(nn.Module):
    """
    입력:
      x: (B, F, T)

    출력:
      y_hat: (B, H)
    """

    def __init__(
        self,
        num_features: int,
        prediction_length: int,
        hidden_channels: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.1,
        use_input_projection: bool = True,
    ):
        super().__init__()
        padding = kernel_size // 2

        if use_input_projection:
            self.input_proj = nn.Conv1d(num_features, hidden_channels, kernel_size=1)
            in_channels = hidden_channels
        else:
            self.input_proj = nn.Identity()
            in_channels = num_features

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels),
            nn.Dropout(dropout),

            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels),
            nn.Dropout(dropout),

            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels),
            nn.Dropout(dropout),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, prediction_length),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        h = self.feature_extractor(x)
        y_hat = self.head(h)
        return y_hat


class CNN1DLightningModule(pl.LightningModule):
    def __init__(
        self,
        num_features: int,
        prediction_length: int,
        hidden_channels: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.1,
        use_input_projection: bool = True,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        loss_name: str = "mae",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = CNN1DForecaster(
            num_features=num_features,
            prediction_length=prediction_length,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size,
            dropout=dropout,
            use_input_projection=use_input_projection,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _compute_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.hparams.loss_name == "mse":
            return F.mse_loss(y_hat, y)
        return F.l1_loss(y_hat, y)

    def _shared_step(self, batch, stage: str):
        x, y, meta = batch
        y_hat = self(x)

        loss = self._compute_loss(y_hat, y)
        mae = F.l1_loss(y_hat, y)
        mse = F.mse_loss(y_hat, y)
        rmse = torch.sqrt(mse)

        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_mae", mae, prog_bar=True, on_step=False, on_epoch=True)
        self.log(f"{stage}_rmse", rmse, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0) -> Dict[str, Any]:
        x, y, meta = batch
        y_hat = self(x)
        return {
            "y_hat": y_hat.detach().cpu(),
            "y": y.detach().cpu(),
            "meta": meta,
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }