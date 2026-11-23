from __future__ import annotations

from typing import Any, Dict, Optional, List

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1DQuantileStyle(nn.Module):
    """
    Fig 스타일 1D CNN + Quantile output

    기본 구조:
      Conv1D -> MaxPool1D(or AdaptivePool) -> Flatten
      -> Dense(64) -> Dense(8) -> Dense(output_dim * num_quantiles)

    입력:
      x: (B, F, T)

    출력:
      y_hat: (B, H, Q)
        - H = output_dim (예측 horizon)
        - Q = len(quantiles)

    예:
      output_dim=11, quantiles=[0.1, 0.5, 0.9]
      -> 출력 shape = (B, 11, 3)
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
        quantiles: Optional[List[float]] = None,
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
        self.quantiles = quantiles or [0.1, 0.5, 0.9]
        self.num_quantiles = len(self.quantiles)

        if activation == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU()

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

        if use_adaptive_pool:
            self.pool = nn.AdaptiveMaxPool1d(1)
        else:
            self.pool = nn.MaxPool1d(kernel_size=conv_out_len)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(conv_out_channels, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, output_dim * self.num_quantiles)

        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, F, T)
        returns:
          y_hat: (B, H, Q)
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

        y_hat = self.fc3(x)        # (B, H*Q)
        y_hat = y_hat.view(x.size(0), self.output_dim, self.num_quantiles)  # (B, H, Q)
        return y_hat


class CNN1DQuantileLightningModule(pl.LightningModule):
    """
    Lightning wrapper for paper-style CNN1D baseline with quantile forecasting.

    batch:
      x: (B, F, T)
      y: (B, H) or (B, 1)

    model output:
      y_hat: (B, H, Q)
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
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        loss_name: str = "quantile",   # "quantile", "mae", "mse"
        optimizer_name: str = "adam",
        quantiles: Optional[List[float]] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.quantiles = quantiles or [0.1, 0.5, 0.9]

        self.model = CNN1DQuantileStyle(
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
            quantiles=self.quantiles,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def quantile_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        y_hat: (B, H, Q)
        y:     (B, H) or (B, 1)
        """
        if y.ndim == 1:
            y = y.unsqueeze(-1)  # (B,1)
        if y.ndim == 2:
            y = y.unsqueeze(-1)  # (B,H,1)

        qs = y_hat.new_tensor(self.quantiles).view(1, 1, -1)   # (1,1,Q)
        err = y - y_hat                                         # (B,H,Q)
        loss = torch.maximum(qs * err, (qs - 1.0) * err)
        return loss.mean()

    def _get_p50(self, y_hat: torch.Tensor) -> torch.Tensor:
        """
        y_hat: (B,H,Q) -> p50: (B,H)
        """
        if 0.5 not in self.quantiles:
            raise ValueError(
                f"0.5 quantile not found in quantiles={self.quantiles}. "
                "P50 metric 계산을 위해 0.5를 포함해야 함."
            )
        p50_idx = self.quantiles.index(0.5)
        return y_hat[:, :, p50_idx]

    def _compute_loss(self, y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        loss_name = self.hparams.loss_name.lower()

        if y.ndim == 1:
            y = y.unsqueeze(-1)

        if loss_name == "quantile":
            return self.quantile_loss(y_hat, y)

        y_p50 = self._get_p50(y_hat)

        if loss_name == "mse":
            return F.mse_loss(y_p50, y)

        return F.l1_loss(y_p50, y)

    def _shared_step(self, batch, stage: str):
        x, y, meta = batch
        y_hat = self(x)  # (B,H,Q)

        if y.ndim == 1:
            y = y.unsqueeze(-1)

        loss = self._compute_loss(y_hat, y)

        # point metric은 보통 p50 기준
        y_p50 = self._get_p50(y_hat)

        mae = F.l1_loss(y_p50, y)
        mse = F.mse_loss(y_p50, y)
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
        y_hat = self(x)  # (B,H,Q)

        if y.ndim == 1:
            y = y.unsqueeze(-1)

        out = {
            "y_hat": y_hat.detach().cpu(),  # (B,H,Q)
            "y": y.detach().cpu(),
            "meta": meta,
        }

        # 편의상 quantile별 분리 결과도 같이 넣음
        for i, q in enumerate(self.quantiles):
            q_name = str(q).replace(".", "")
            out[f"pred_q{q_name}"] = y_hat[:, :, i].detach().cpu()

        return out

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
    # 예시:
    # 입력 (B, F, T) = (4, 6, 24)
    # 출력 horizon = 11
    # quantiles = [0.1, 0.5, 0.9]
    model = CNN1DQuantileStyle(
        num_features=6,
        encoder_length=24,
        output_dim=11,
        conv_out_channels=64,
        kernel_size=3,
        hidden_dim1=64,
        hidden_dim2=8,
        dropout=0.0,
        use_adaptive_pool=False,
        quantiles=[0.1, 0.5, 0.9],
    )
    x = torch.randn(4, 6, 24)
    y = model(x)

    print("output shape:", y.shape)   # 기대: (4, 11, 3)
    print("trainable params:", count_parameters(model))