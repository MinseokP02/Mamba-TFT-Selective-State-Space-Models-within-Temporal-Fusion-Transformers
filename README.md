# Environment Setup
Requires Python 3.10. `mamba_ssm` and `causal_conv1d` are built against CUDA, so install PyTorch (CUDA 12.1) first, then install the remaining packages.
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121 && pip install -r requirements.txt --no-build-isolation
```

## Grid-connected PV
### Train
```bash
python run.py train --config configs/tft_all.yaml
```
### Predict
```bash
python run.py predict --config configs/tft_all.yaml --ckpt <checkpoint_path>
```
For `<checkpoint_path>`, use the `best_ckpt` path recorded in `meta` of the `reports/run_...` directory created by training.
### Test
```bash
python run.py eval --config configs/tft_all.yaml --run_dir <run_directory>
```
For `<run_directory>`, use the absolute path of the `reports/run_..._pred` directory created by prediction.
### One Shot
```bash
python run.py all --config configs/tft_all.yaml
```

## Stand-alone PV
### Train
```bash
python run.py train --config configs/tft_independence.yaml
```
### Predict
```bash
python run.py predict --config configs/tft_independence.yaml --ckpt <checkpoint_path>
```
Same as above.
### Test
```bash
python run.py eval --config configs/tft_independence.yaml --run_dir <run_directory>
```
Same as above.
### One Shot
```bash
python run.py all --config configs/tft_independence.yaml
```

## Utrecht ablation

The command below prints the default experiment plan for LSTM, CNN, TFT, Mamba, and Mamba-TFT using the Utrecht-specific configs.

```bash
python run.py ablation --models all
```

Training starts only when `--run` is added. When several models are selected at once, shared parameters can be used.

```bash
python run.py ablation --models all \
  --dropout 0.1,0.3,0.5 --batch-size 64,128 --repeat 3 --max-runs 100 --run
```

Model-specific parameters can also be searched with the same runner.

```bash
python run.py ablation --models lstm --hidden-size 32,64 --num-layers 1,2 --run
python run.py ablation --models cnn --conv-out-channels 32,64 --kernel-size 3,5 --run
python run.py ablation --models tft --hidden-size 32,64 --attention-head-size 2,4 --run
python run.py ablation --models mamba_tft --mamba-layers 1,2,3 --mamba-d-state 8,16 --run
```

Select the Osong dataset with `--dataset osong`. The default Osong config for each model is applied automatically.

```bash
python run.py ablation --dataset osong --models simple_mamba --mode grid \
  --learning-rate 0.01,0.001,0.0001 --dropout 0.1,0.2,0.4,0.6 \
  --hidden-size 32,64 --mamba-layers 1,2 --mamba-d-state 8,16 \
  --mamba-d-conv 2,4 --mamba-expand 1,2 --max-runs 384 --run
```

The default search mode is one-factor; use `--mode grid` to run every combination. The existing `mt-ablation --config ...` command is still supported.
