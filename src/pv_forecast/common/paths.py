from pathlib import Path
from datetime import datetime
import json, shutil

def make_run_dir(base_dir: str, name: str) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base_dir) / f"run_{ts}_{name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "tables").mkdir()
    (run_dir / "figures").mkdir()
    (run_dir / "configs").mkdir()
    (run_dir / "meta").mkdir()
    return run_dir

def copy_config(cfg_path: str, run_dir: Path):
    shutil.copy(cfg_path, run_dir / "configs" / Path(cfg_path).name)

def save_json(obj: dict, path: Path):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")