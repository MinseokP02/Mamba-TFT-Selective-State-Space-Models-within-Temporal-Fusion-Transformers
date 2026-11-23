# 가상환경 다운로드
Python 3.10 기준. mamba_ssm·causal_conv1d는 CUDA 빌드가 필요하므로 torch(CUDA 12.1)를 먼저 설치한 뒤 나머지를 설치합니다.
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121 && pip install -r requirements.txt --no-build-isolation
```
## 계통 연계형
## Train
python run.py train --config configs/tft_all.yaml
## Predict
python run.py predict --config configs/tft_all.yaml --ckpt <checkpoint_path>
checkpoint_path의 경우, train 이후 생긴 reports/run... 에서 meta의 best_ckpt 경로 입력 필요
## Test
python run.py eval --config configs/tft_all.yaml --run_dir <run_directory>
run_directory의 겨우, predict 이후 생긴 reports/run_..._pred 의 절대 경로 입력 필요
## One Shot
python run.py all --config configs/tft_all.yaml

## 독립형
## Train
python run.py train --config configs/tft_independence.yaml
## Predict
python run.py predict --config configs/tft_independence.yaml --ckpt <checkpoint_path>
위와 동일
## Test
python run.py eval --config configs/tft_all.yaml --run_dir <run_directory>
위와 동일
## One Shot
python run.py all --config configs/tft_independence.yaml

## Utrecht ablation

아래 명령은 Utrecht 전용 config로 LSTM, CNN, TFT, Mamba-TFT의 기본 실험 계획을 확인합니다.

```bash
python run.py ablation --models all
```

`--run`을 붙여야 실제 학습을 시작합니다. 여러 모델을 한 번에 선택할 때는 공통 파라미터를 사용할 수 있습니다.

```bash
python run.py ablation --models all \
  --dropout 0.1,0.3,0.5 --batch-size 64,128 --repeat 3 --max-runs 100 --run
```

모델별 파라미터도 같은 runner에서 탐색할 수 있습니다.

```bash
python run.py ablation --models lstm --hidden-size 32,64 --num-layers 1,2 --run
python run.py ablation --models cnn --conv-out-channels 32,64 --kernel-size 3,5 --run
python run.py ablation --models tft --hidden-size 32,64 --attention-head-size 2,4 --run
python run.py ablation --models mamba_tft --mamba-layers 1,2,3 --mamba-d-state 8,16 --run
```

오송 데이터셋은 `--dataset osong`으로 선택합니다. 모델별 오송 기본 config가
자동으로 적용됩니다.

```bash
python run.py ablation --dataset osong --models simple_mamba --mode grid \
  --learning-rate 0.01,0.001,0.0001 --dropout 0.1,0.2,0.4,0.6 \
  --hidden-size 32,64 --mamba-layers 1,2 --mamba-d-state 8,16 \
  --mamba-d-conv 2,4 --mamba-expand 1,2 --max-runs 384 --run
```

기본값은 one-factor 방식이며, 모든 조합은 `--mode grid`로 실행합니다. 기존
`mt-ablation --config ...` 명령도 그대로 지원합니다.
