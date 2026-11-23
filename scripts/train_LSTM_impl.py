import os
import yaml
import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from pv_forecast.common.seed import seed_all
from pv_forecast.common.paths import make_run_dir, copy_config, save_json
from pv_forecast.data.build_df_full import build_df_full

class LSTMDataset(Dataset):
    def __init__(self, data, target, window_size, horizon):
        self.data = torch.FloatTensor(data)
        self.target = torch.FloatTensor(target)
        self.window_size = window_size
        self.horizon = horizon

    def __len__(self):
        return len(self.data) - self.window_size - self.horizon + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.window_size]
        y = self.target[idx + self.window_size : idx + self.window_size + self.horizon]
        return x, y

class LSTMForecaster(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_dim, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.quantiles = [0.1, 0.5, 0.9]
        self.output_dim = output_dim
        self.num_quantiles = len(self.quantiles)
        self.fc = nn.Linear(hidden_size, output_dim*self.num_quantiles) 
        

    def forward(self, x):
        out, _ = self.lstm(x)
        y_hat = self.fc(out[:, -1, :])
        y_hat = y_hat.view(x.size(0), self.output_dim, self.num_quantiles)
        return y_hat
    
    def pinball_loss(self, y_hat, y):
        """
        y_hat: (B, H, Q)
        y:     (B, H)
        """
        if y.ndim == 1:
            y = y.unsqueeze(-1)

        if y_hat.ndim != 3:
            raise ValueError(f"Expected y_hat shape (B,H,Q), got {tuple(y_hat.shape)}")

        if y.ndim != 2:
            raise ValueError(f"Expected y shape (B,H), got {tuple(y.shape)}")

        if y_hat.size(1) != y.size(1):
            raise ValueError(
                f"Horizon mismatch: y_hat H={y_hat.size(1)}, y H={y.size(1)}"
            )

        qs = y_hat.new_tensor(self.quantiles).view(1, 1, -1)
        y = y.unsqueeze(-1)

        errors = y - y_hat
        loss = torch.maximum(qs * errors, (qs - 1.0) * errors)
        return loss.mean()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.pinball_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.pinball_loss(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

def main(config_path: str):
    cfg = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
    seed_all(cfg["run"]["seed"])

    run_dir = make_run_dir(cfg["run"]["reports_dir"], cfg["run"]["name"])
    copy_config(config_path, run_dir)

    df_full = build_df_full(cfg["data"])
    n = len(df_full)

    # 1. TFT 방식의 Split 로직 적용
    train_ratio = cfg["split"]["train_ratio"]
    val_ratio = cfg["split"]["val_ratio"]
    test_ratio = cfg["split"].get("test_ratio", 1.0 - train_ratio - val_ratio)

    if not (0 < train_ratio < 1): raise ValueError(f"train_ratio error: {train_ratio}")
    if test_ratio <= 0: raise ValueError("Sum of ratios must be < 1.0 to have test set.")

    train_cutoff = int(n * train_ratio)
    val_cutoff = int(n * (train_ratio + val_ratio))

    df_train = df_full.iloc[:train_cutoff].copy()
    df_val = df_full.iloc[train_cutoff:val_cutoff].copy()
    df_test = df_full.iloc[val_cutoff:].copy()

    # 2. 기능 및 타겟 정의
    features = cfg["features"]["input_reals"]
    target = cfg["features"]["target"]

    # 3. 스케일링 (Train 기준으로 Fit)
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()
    
    df_train[features] = scaler_x.fit_transform(df_train[features])
    df_val[features] = scaler_x.transform(df_val[features])
    df_test[features] = scaler_x.transform(df_test[features])

    df_train[target] = scaler_y.fit_transform(df_train[[target]])
    df_val[target] = scaler_y.transform(df_val[[target]])
    df_test[target] = scaler_y.transform(df_test[[target]])

    # 4. 데이터셋 생성 (Window 적용)
    window_size = cfg["lstm"]["window_size"]
    horizon = cfg["lstm"]["horizon"]

    train_ds = LSTMDataset(df_train[features].values, df_train[target].values, window_size, horizon)
    val_ds = LSTMDataset(df_val[features].values, df_val[target].values, window_size, horizon)
    test_ds = LSTMDataset(df_test[features].values, df_test[target].values, window_size, horizon)

    # 데이터셋 비어있는지 검증
    if len(val_ds) <= 0: raise ValueError("Validation dataset is empty.")
    if len(test_ds) <= 0: raise ValueError("Test dataset is empty.")

    # 5. 데이터로더
    train_loader = DataLoader(train_ds, batch_size=cfg["lstm"]["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=cfg["lstm"]["batch_size"], shuffle=False, num_workers=0)
    #test_loader = DataLoader(test_ds, batch_size=cfg["lstm"]["batch_size"], shuffle=False, num_workers=0)

    # 6. 모델 생성
    model = LSTMForecaster(
        input_size=len(features),
        hidden_size=cfg["lstm"]["hidden_size"],
        num_layers=cfg["lstm"]["num_layers"],
        dropout=cfg["lstm"]["dropout"],
        output_dim=cfg["lstm"]["horizon"],
        lr=cfg["lstm"]["learning_rate"]
    )

    # 7. 체크포인트 설정
    ckpt_dir = f'{cfg["run"]["artifacts_dir"]}/{cfg["run"]["name"]}'
    ckpt_cb = ModelCheckpoint(
        dirpath=ckpt_dir, 
        save_top_k=1, 
        monitor="val_loss", 
        mode="min", 
        filename="lstm-quantile-{epoch}-{val_loss:.4f}"
    )

    # 8. 트레이너 실행
    trainer = pl.Trainer(
        max_epochs=cfg["lstm"]["max_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=5, mode="min"),
            LearningRateMonitor(logging_interval="epoch"),
            ckpt_cb
        ],
        default_root_dir=str(run_dir),
    )

    trainer.fit(model, train_loader, val_loader)

    # 9. 메타 정보 저장
    best_ckpt = ckpt_cb.best_model_path
    save_json(
        {
            "best_ckpt": best_ckpt, 
            "run_dir": str(run_dir), 
            "train_cutoff": train_cutoff,
            "val_cutoff": val_cutoff
        },
        run_dir / "meta" / "train_meta.json"
    )

    print(f"RUN_DIR: {run_dir}")
    print(f"BEST_CKPT: {best_ckpt}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    main(args.config)