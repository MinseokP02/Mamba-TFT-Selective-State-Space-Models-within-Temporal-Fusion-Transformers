#!/usr/bin/env bash
set -euo pipefail

PYTHON=/home/bml/miniforge3/envs/solar_tft/bin/python
CONFIG=configs/mamba_tft_utrecht_weather_single_id.yaml

# Missing partition 1: batch sizes 64 and 128, every LR/dropout/encoder value.
"$PYTHON" run.py mt-ablation \
  --models mamba_tft \
  --config "$CONFIG" \
  --mode grid \
  --learning-rate 0.01,0.001,0.0001 \
  --dropout 0.1,0.2,0.4,0.6 \
  --batch-size 64,128 \
  --max-encoder-length 33,55,77,99,154 \
  --max-runs 120 \
  --run

# Missing partition 2: batch size 32 with encoder lengths other than 77.
"$PYTHON" run.py mt-ablation \
  --models mamba_tft \
  --config "$CONFIG" \
  --mode grid \
  --learning-rate 0.01,0.001,0.0001 \
  --dropout 0.1,0.2,0.4,0.6 \
  --batch-size 32 \
  --max-encoder-length 33,55,99,154 \
  --max-runs 48 \
  --run

# Missing partition 3: batch 32 / encoder 77 / LR 0.01.
"$PYTHON" run.py mt-ablation \
  --models mamba_tft \
  --config "$CONFIG" \
  --mode grid \
  --learning-rate 0.01 \
  --dropout 0.1,0.2,0.6 \
  --batch-size 32 \
  --max-encoder-length 77 \
  --max-runs 3 \
  --run

# Missing partition 4: batch 32 / encoder 77 / LR 0.0001.
"$PYTHON" run.py mt-ablation \
  --models mamba_tft \
  --config "$CONFIG" \
  --mode grid \
  --learning-rate 0.0001 \
  --dropout 0.1,0.2,0.6 \
  --batch-size 32 \
  --max-encoder-length 77 \
  --max-runs 3 \
  --run
