# src/pv_forecast/models/cnn1d_module.py

from __future__ import annotations

from typing import Any, Dict, Optional

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1DPaperStyle(nn.Module):
    """
    Fig 스타일 1D CNN

    기본 구조:
      Conv1D -> MaxPool1D -> Flatten -> Dense(64) -> Dense(8) -> Dense(output_dim)

    입력:
      x: (B, F, T)

    출력:
      y_hat: (B, output_dim)

    참고:
      - 논문 그림은 사실상 in_channels=1, seq_len=5, output_dim=1 에 가까움
      - 여기서는 pv_forecast 실험용으로 일반화함
    """

    def __init__(
        self,
        num_features: int,
        encoder_length: int,
        output_dim: int = 1,
        conv_out_channels: int = 64,
        kernel_size: int = 3,
        hidden_dim1: int = 64,
        hidden_dim2: int = 8,
        activation: str = "relu",
        dropout: float = 0.0,
        use_adaptive_pool: bool = False,
    ):
        super().__init__()

        if encoder_length < kernel_size:
            raise ValueError(
                f"encoder_length({encoder_length}) must be >= kernel_size({kernel_size})"
            )

        self.num_features = num_features
        self.encoder_length = encoder_length
        self.output_dim = output_dim
        self.conv_out_channels = conv_out_channels
        self.kernel_size = kernel_size
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.dropout = dropout
        self.use_adaptive_pool = use_adaptive_pool

        if activation == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU()

        # 논문 그림 스타일: valid conv
        self.conv1 = nn.Conv1d(
            in_channels=num_features,
            out_channels=conv_out_channels,
            kernel_size=kernel_size,
            padding=0,
        )

        conv_out_len = encoder_length - kernel_size + 1
        if conv_out_len <= 0:
            raise ValueError(
                f"Invalid conv output length: {conv_out_len}. "
                f"Check encoder_length={encoder_length}, kernel_size={kernel_size}"
            )

        # 그림 스타일이면 보통 conv 결과 전체를 한 번에 pooling해서 길이 1로 만듦
        if use_adaptive_pool:
            self.pool = nn.AdaptiveMaxPool1d(1)
        else:
            self.pool = nn.MaxPool1d(kernel_size=conv_out_len)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(conv_out_channels, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, F, T)
        """
        x = self.conv1(x)          # (B, C, T')
        x = self.pool(x)           # (B, C, 1)
        x = self.flatten(x)        # (B, C)

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.act(x)
        x = self.drop(x)

        y_hat = self.fc3(x)        # (B, output_dim)
        return y_hat


class CNN1DLightningModule(pl.LightningModule):
    """
    Lightning wrapper for paper-style CNN1D baseline.

    batch:
      x: (B, F, T)
      y: (B, H) or (B, 1)

    model output:
      y_hat: (B, output_dim)
    """

    def __init__(
        self,
        num_features: int,
        encoder_length: int,
        output_dim: int,
        conv_out_channels: int = 64,
        kernel_size: int = 3,
        hidden_dim1: int = 64,
        hidden_dim2: int = 8,
        activation: str = "relu",
        dropout: float = 0.0,
        use_adaptive_pool: bool = False,
        quantiles: Optional[list[float]] = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        loss_name: str = "mae",
        optimizer_name: str = "adam",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = CNN1DPaperStyle(
            num_features=num_features,
            encoder_length=encoder_length,
            output_dim=output_dim,
            conv_out_channels=conv_out_channels,
            kernel_size=kernel_size,
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2,
            activation=activation,
            dropout=dropout,
            use_adaptive_pool=use_adaptive_pool,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _compute_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.hparams.loss_name.lower() == "mse":
            return F.mse_loss(y_hat, y)
        return F.l1_loss(y_hat, y)

    def _shared_step(self, batch, stage: str):
        x, y, meta = batch
        y_hat = self(x)

        # shape 방어
        if y_hat.ndim == 1:
            y_hat = y_hat.unsqueeze(-1)
        if y.ndim == 1:
            y = y.unsqueeze(-1)

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

        if y_hat.ndim == 1:
            y_hat = y_hat.unsqueeze(-1)
        if y.ndim == 1:
            y = y.unsqueeze(-1)

        return {
            "y_hat": y_hat.detach().cpu(),
            "y": y.detach().cpu(),
            "meta": meta,
        }

    def configure_optimizers(self):
        opt_name = self.hparams.optimizer_name.lower()

        if opt_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.lr,
                weight_decay=self.hparams.weight_decay,
            )
        else:
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


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 예시 1: 논문 그림과 유사한 구조
    # 입력 (B, 1, 5), 출력 (B, 1)
    model = CNN1DPaperStyle(
        num_features=1,
        encoder_length=5,
        output_dim=1,
        conv_out_channels=64,
        kernel_size=3,
        hidden_dim1=64,
        hidden_dim2=8,
        dropout=0.0,
        use_adaptive_pool=False,
    )
    x = torch.randn(4, 1, 5)
    y = model(x)
    print("output shape:", y.shape)
    print("trainable params:", count_parameters(model))
    # 기대: 4945