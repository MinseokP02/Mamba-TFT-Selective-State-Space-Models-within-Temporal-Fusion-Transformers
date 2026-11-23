import os, random
import numpy as np
import torch
import lightning.pytorch as pl

def seed_all(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # GPU 재현성(속도 조금 희생)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Lightning 쪽까지(데이터로더 워커까지)
    pl.seed_everything(seed, workers=True)

    torch.use_deterministic_algorithms(True, warn_only=True)
