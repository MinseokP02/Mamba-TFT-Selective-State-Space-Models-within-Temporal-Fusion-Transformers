import argparse
from pv_forecast.scripts_like.train_impl import train_main
from pv_forecast.scripts_like.predict_impl import predict_main
from pv_forecast.scripts_like.eval_impl import eval_main

def main():
    ap = argparse.ArgumentParser(prog="pv_forecast")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("train")
    p1.add_argument("--config", required=True)

    p2 = sub.add_parser("predict")
    p2.add_argument("--config", required=True)
    p2.add_argument("--ckpt", required=True)

    p3 = sub.add_parser("eval")
    p3.add_argument("--config", required=True)
    p3.add_argument("--run_dir", required=True)

    args = ap.parse_args()

    if args.cmd == "train":
        train_main(args.config)
    elif args.cmd == "predict":
        predict_main(args.config, args.ckpt)
    elif args.cmd == "eval":
        eval_main(args.config, args.run_dir)