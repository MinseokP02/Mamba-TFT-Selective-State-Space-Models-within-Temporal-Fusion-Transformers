# src/pv_forecast/datasets/seq_dataset.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

import lightning.pytorch as pl


@dataclass
class SeqSample:
    x: np.ndarray
    y: np.ndarray
    group_id: str
    encoder_time_idx_start: int
    encoder_time_idx_end: int
    decoder_time_idx_start: int
    decoder_time_idx_end: int


class SequenceDataset(Dataset):
    """
    TFT와 동일한 time_idx 기반 분할 기준을 따르는 sequence dataset.

    split 기준:
      - train: decoder 마지막 time_idx <= training_cutoff
      - val  : training_cutoff < decoder 마지막 time_idx <= validation_cutoff
      - test : validation_cutoff < decoder 마지막 time_idx

    return:
      - x: (F, T) if model_type='cnn1d'
           (T, F) if model_type='lstm'
      - y: (H,)
      - meta: dict
    """

    def __init__(
        self,
        df_full: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        group_col: str = "series_id",
        time_idx_col: str = "time_idx",
        encoder_length: int = 168,
        prediction_length: int = 24,
        split: str = "train",
        training_cutoff: Optional[int] = None,
        validation_cutoff: Optional[int] = None,
        test_start_time_idx: Optional[int] = None,
        allow_missing_time_idx: bool = False,
        model_type: str = "cnn1d",
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        assert split in ["train", "val", "test"]
        assert model_type in ["cnn1d", "lstm"]

        if training_cutoff is None or validation_cutoff is None:
            raise ValueError("training_cutoff and validation_cutoff must be provided")
        if training_cutoff >= validation_cutoff:
            raise ValueError("training_cutoff must be smaller than validation_cutoff")

        self.feature_cols = feature_cols
        self.target_col = target_col
        self.group_col = group_col
        self.time_idx_col = time_idx_col
        self.encoder_length = encoder_length
        self.prediction_length = prediction_length
        self.split = split
        self.training_cutoff = training_cutoff
        self.validation_cutoff = validation_cutoff
        self.test_start_time_idx = test_start_time_idx
        self.allow_missing_time_idx = allow_missing_time_idx
        self.model_type = model_type
        self.dtype = dtype

        required_cols = [group_col, time_idx_col, target_col] + feature_cols
        missing = [c for c in required_cols if c not in df_full.columns]
        if missing:
            raise ValueError(f"Missing columns in df_full: {missing}")

        self.df_full = df_full[required_cols].copy()
        self.df_full = self.df_full.sort_values([group_col, time_idx_col]).reset_index(drop=True)

        self.samples: List[SeqSample] = []
        self._build_samples()

    def _is_contiguous_time_idx(self, arr: np.ndarray) -> bool:
        if len(arr) <= 1:
            return True
        return np.all(np.diff(arr) == 1)

    def _match_split(self, decoder_start_t: int, decoder_end_t: int) -> bool:
        if self.split == "train":
            # TFT의 df_train은 train_cutoff_time_idx 이전까지만 포함
            return decoder_end_t < self.training_cutoff

        if self.split == "val":
            # prediction 시작은 train_cutoff 이후, target 끝은 val_cutoff 이전
            return (
                decoder_start_t >= self.training_cutoff
                and decoder_end_t < self.validation_cutoff
            )

        if self.split == "test":
            # TFT의 min_prediction_idx=val_cutoff_time_idx와 동일한 의미
            start = self.test_start_time_idx
            if start is None:
                start = self.validation_cutoff
            return decoder_start_t >= start

        return False

    def _build_samples(self) -> None:
        total_window = self.encoder_length + self.prediction_length

        for group_id, gdf in self.df_full.groupby(self.group_col):
            gdf = gdf.sort_values(self.time_idx_col).reset_index(drop=True)

            time_idx_arr = gdf[self.time_idx_col].to_numpy()
            feature_arr = gdf[self.feature_cols].to_numpy(dtype=np.float32)
            target_arr = gdf[self.target_col].to_numpy(dtype=np.float32)

            if len(gdf) < total_window:
                continue

            for start in range(0, len(gdf) - total_window + 1):
                end = start + total_window
                window_time_idx = time_idx_arr[start:end]

                if not self.allow_missing_time_idx and not self._is_contiguous_time_idx(window_time_idx):
                    continue

                enc_start = start
                enc_end = start + self.encoder_length
                dec_end = end

                decoder_start_t = int(time_idx_arr[enc_end])
                decoder_end_t = int(time_idx_arr[dec_end - 1])

                if not self._match_split(decoder_start_t, decoder_end_t):
                    continue

                x = feature_arr[enc_start:enc_end]   # (T, F)
                y = target_arr[enc_end:dec_end]      # (H,)

                if self.model_type == "cnn1d":
                    x = x.T  # (F, T)
                # lstm이면 그대로 (T, F)

                self.samples.append(
                    SeqSample(
                        x=x,
                        y=y,
                        group_id=str(group_id),
                        encoder_time_idx_start=int(time_idx_arr[enc_start]),
                        encoder_time_idx_end=int(time_idx_arr[enc_end - 1]),
                        decoder_time_idx_start=int(time_idx_arr[enc_end]),
                        decoder_time_idx_end=int(time_idx_arr[dec_end - 1]),
                    )
                )

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No samples created for split={self.split}. "
                f"Check cutoffs, encoder_length, prediction_length, and time_idx continuity."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]

        x = torch.tensor(item.x, dtype=self.dtype)
        y = torch.tensor(item.y, dtype=self.dtype)

        meta = {
            "group_id": item.group_id,
            "encoder_time_idx_start": item.encoder_time_idx_start,
            "encoder_time_idx_end": item.encoder_time_idx_end,
            "decoder_time_idx_start": item.decoder_time_idx_start,
            "decoder_time_idx_end": item.decoder_time_idx_end,
        }
        return x, y, meta


class SequenceDataModule(pl.LightningDataModule):
    def __init__(
        self,
        df_full: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        group_col: str = "series_id",
        time_idx_col: str = "time_idx",
        encoder_length: int = 168,
        prediction_length: int = 24,
        training_cutoff: Optional[int] = None,
        validation_cutoff: Optional[int] = None,
        test_start_time_idx: Optional[int] = None,
        batch_size: int = 64,
        num_workers: int = 0,
        allow_missing_time_idx: bool = False,
        model_type: str = "cnn1d",
    ):
        super().__init__()
        self.df_full = df_full
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.group_col = group_col
        self.time_idx_col = time_idx_col
        self.encoder_length = encoder_length
        self.prediction_length = prediction_length
        self.training_cutoff = training_cutoff
        self.validation_cutoff = validation_cutoff
        self.test_start_time_idx = test_start_time_idx
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.allow_missing_time_idx = allow_missing_time_idx
        self.model_type = model_type

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        if stage in (None, "fit"):
            self.train_dataset = SequenceDataset(
                df_full=self.df_full,
                feature_cols=self.feature_cols,
                target_col=self.target_col,
                group_col=self.group_col,
                time_idx_col=self.time_idx_col,
                encoder_length=self.encoder_length,
                prediction_length=self.prediction_length,
                split="train",
                training_cutoff=self.training_cutoff,
                validation_cutoff=self.validation_cutoff,
                allow_missing_time_idx=self.allow_missing_time_idx,
                model_type=self.model_type,
            )

            self.val_dataset = SequenceDataset(
                df_full=self.df_full,
                feature_cols=self.feature_cols,
                target_col=self.target_col,
                group_col=self.group_col,
                time_idx_col=self.time_idx_col,
                encoder_length=self.encoder_length,
                prediction_length=self.prediction_length,
                split="val",
                training_cutoff=self.training_cutoff,
                validation_cutoff=self.validation_cutoff,
                allow_missing_time_idx=self.allow_missing_time_idx,
                model_type=self.model_type,
            )

        if stage in ("test", "predict"):
            self.test_dataset = SequenceDataset(
                df_full=self.df_full,
                feature_cols=self.feature_cols,
                target_col=self.target_col,
                group_col=self.group_col,
                time_idx_col=self.time_idx_col,
                encoder_length=self.encoder_length,
                prediction_length=self.prediction_length,
                split="test",
                training_cutoff=self.training_cutoff,
                validation_cutoff=self.validation_cutoff,
                test_start_time_idx=self.test_start_time_idx,
                allow_missing_time_idx=self.allow_missing_time_idx,
                model_type=self.model_type,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )