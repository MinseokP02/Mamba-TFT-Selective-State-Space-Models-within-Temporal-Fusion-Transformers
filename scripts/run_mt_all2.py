from __future__ import annotations

import argparse

from scripts.run_all3 import add_cyclical_time_features
from scripts.run_mt_all import main as run_mt_all_main


def main(config_path: str) -> None:
    """Run Mamba-TFT with sine/cosine calendar covariates."""
    run_mt_all_main(
        config_path,
        dataframe_transform=add_cyclical_time_features,
        run_suffix="_all2",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
