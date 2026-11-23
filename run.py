import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import argparse

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("train")
    p1.add_argument("--config", required=True)

    p2 = sub.add_parser("predict")
    p2.add_argument("--config", required=True)
    p2.add_argument("--ckpt", required=True)

    p3 = sub.add_parser("eval")
    p3.add_argument("--config", required=True)
    p3.add_argument("--run_dir", required=True)

    p4 = sub.add_parser("all")
    p4.add_argument("--config", required=True)

    p4_1 = sub.add_parser("all2")
    p4_1.add_argument("--config", required=True)

    p4_2 = sub.add_parser("all3")
    p4_2.add_argument("--config", required=True)

    # CNN1D
    p5 = sub.add_parser("train-cnn")
    p5.add_argument("--config", required=True)

    p6 = sub.add_parser("predict-cnn")
    p6.add_argument("--config", required=True)
    p6.add_argument("--ckpt", required=True)

    p7 = sub.add_parser("eval-cnn")
    p7.add_argument("--config", required=True)
    p7.add_argument("--run_dir", required=True)

    p8 = sub.add_parser("all-cnn")
    p8.add_argument("--config", required=True)

    p8_1 = sub.add_parser("all-cnn2")
    p8_1.add_argument("--config", required=True)

    p8_2 = sub.add_parser("all-cnn3")
    p8_2.add_argument("--config", required=True)

    # --- LSTM Commands ---
    p9 = sub.add_parser("train-lstm")
    p9.add_argument("--config", required=True)

    p10 = sub.add_parser("predict-lstm")
    p10.add_argument("--config", required=True)
    p10.add_argument("--ckpt", required=True)

    p11 = sub.add_parser("eval-lstm")
    p11.add_argument("--config", required=True)
    p11.add_argument("--run_dir", required=True)

    p9_1 = sub.add_parser("all-lstm")
    p9_1.add_argument("--config", required=True)

    # --- Heatmap Command ---
    p12 = sub.add_parser("heatmap")
    p12.add_argument("--pred_csv", action="append", required=True)
    p12.add_argument("--label", action="append", required=True)
    p12.add_argument("--out_dir", required=True)
    p12.add_argument("--start_hour", type=int, default=7)
    p12.add_argument("--end_hour", type=int, default=17)

    p13 = sub.add_parser("mt_all")
    p13.add_argument("--config", required=True)

    p14 = sub.add_parser("mt_all2")
    p14.add_argument("--config", required=True)

    p15 = sub.add_parser("mt-ablation")
    from scripts.run_mt_ablation import add_arguments as add_mt_ablation_arguments
    add_mt_ablation_arguments(p15)

    p16 = sub.add_parser("ablation")
    add_mt_ablation_arguments(p16)

    p17 = sub.add_parser("validation-ablation")
    from scripts.run_validation_ablation import add_arguments as add_validation_ablation_arguments
    add_validation_ablation_arguments(p17)

    p18 = sub.add_parser("all-simple-mamba")
    p18.add_argument("--config", required=True)

    args = ap.parse_args()

    if args.cmd == "train":
        from scripts.train import main as train_main
        train_main(args.config)
    elif args.cmd == "predict":
        from scripts.predict import main as pred_main
        pred_main(args.config, args.ckpt)
    elif args.cmd == "eval":
        from scripts.evaluate import main as eval_main
        eval_main(args.config, args.run_dir)
    elif args.cmd == "all":
        from scripts.run_all import main as all_main
        all_main(args.config)
    elif args.cmd == "all2":
        from scripts.run_all2 import main as all_main2
        all_main2(args.config)
    elif args.cmd == "all3":
        from scripts.run_all3 import main as all_main3
        all_main3(args.config)

    # CNN1D
    elif args.cmd == "train-cnn":
        from scripts.train_cnn_impl import train_cnn_main
        train_cnn_main(args.config)
    elif args.cmd == "predict-cnn":
        from scripts.predict_cnn_impl import predict_cnn_main
        predict_cnn_main(args.config, args.ckpt)
    elif args.cmd == "eval-cnn":
        from scripts.eval_cnn_impl import eval_cnn_main
        eval_cnn_main(args.config, args.run_dir)
    elif args.cmd == "all-cnn":
        from scripts.run_all_cnn import main
        main(args.config)
    elif args.cmd == "all-cnn2":
        from scripts.run_all_cnn2 import main
        main(args.config)
    elif args.cmd == "all-cnn3":
        from scripts.run_cnn_all3 import main
        main(args.config)
    
    # LSTM 
    elif args.cmd == "train-lstm":
        from scripts.train_LSTM_impl import main as train_lstm_main
        train_lstm_main(args.config)
        
    elif args.cmd == "predict-lstm":
        from scripts.predict_LSTM_impl import main as predict_lstm_main
        predict_lstm_main(args.config, args.ckpt)
        
    elif args.cmd == "eval-lstm":
        from scripts.eval_LSTM_impl import main as eval_lstm_main
        eval_lstm_main(args.config, args.run_dir)

    elif args.cmd == "all-lstm":
        from scripts.run_all_lstm import main as all_lstm_main
        all_lstm_main(args.config)

    elif args.cmd == "heatmap":
        from analysis.heatmap import main as heatmap_main
        heatmap_main(args.pred_csv, args.label, args.out_dir, args.start_hour, args.end_hour)

    # run_mt_all
    elif args.cmd == "mt_all":
        from scripts.run_mt_all import main as run_mt_all_main
        run_mt_all_main(args.config)
    elif args.cmd == "mt_all2":
        from scripts.run_mt_all2 import main as run_mt_all2_main
        run_mt_all2_main(args.config)
    elif args.cmd in {"mt-ablation", "ablation"}:
        from scripts.run_mt_ablation import run_from_args
        run_from_args(args)
    elif args.cmd == "validation-ablation":
        from scripts.run_validation_ablation import run_from_args
        run_from_args(args)
    elif args.cmd == "all-simple-mamba":
        from scripts.run_all_simple_mamba import main as simple_mamba_main
        simple_mamba_main(args.config)


if __name__ == "__main__":
    main()
