import numpy as np
import pandas as pd
import torch

def extract_pred_tensor(output):
    pred_tensor = None

    if isinstance(output, dict):
        pred_tensor = output.get("prediction", output.get("output", None))
    else:
        for attr in ["prediction", "predictions", "output", "y_hat"]:
            if hasattr(output, attr):
                pred_tensor = getattr(output, attr)
                break

    if pred_tensor is None:
        raise RuntimeError("Could not find prediction tensor inside output object.")

    if isinstance(pred_tensor, (list, tuple)):
        pred_tensor = pred_tensor[0]

    if not torch.is_tensor(pred_tensor):
        pred_tensor = torch.as_tensor(pred_tensor)

    return pred_tensor

def postprocess_raw_to_frames(raw, df_full: pd.DataFrame):
    output, x, index, decoder_lengths, y = raw
    pred_tensor = extract_pred_tensor(output)

    dec_ti = x["decoder_time_idx"].detach().cpu().numpy().astype(np.int64)  # (N,H)
    true = x["decoder_target"].detach().cpu().numpy()                       # (N,H)
    pred = pred_tensor.detach().cpu().numpy()                               # (N,H,3)

    time_map = df_full[["time_idx", "timestamp"]].drop_duplicates().set_index("time_idx")["timestamp"]
    N, H = dec_ti.shape
    ts_flat = pd.to_datetime(time_map.loc[dec_ti.reshape(-1)].to_numpy())

    out_all = pd.DataFrame({
        "sample": np.repeat(np.arange(N), H),
        "horizon": np.tile(np.arange(1, H + 1), N),
        "horizon_idx": np.tile(np.arange(1, H + 1), N),
        "timestamp": ts_flat,
        "date": ts_flat.floor("D"),
        "hour": ts_flat.hour,
        "true": true.reshape(-1),
        "pred_q10": pred[:, :, 0].reshape(-1),
        "pred_q50": pred[:, :, 1].reshape(-1),
        "pred_q90": pred[:, :, 2].reshape(-1),
    })

    out_all["err"] = out_all["pred_q50"] - out_all["true"]
    out_all["abs_err"] = out_all["err"].abs()
    out_all["sq_err"] = out_all["err"] ** 2

    final_df = out_all.sort_values(["date", "timestamp", "sample"]).reset_index(drop=True)

    one = (final_df
        .groupby("timestamp", as_index=False)
        .agg(true=("true","first"),
             pred_q10=("pred_q10","mean"),
             pred_q50=("pred_q50","mean"),
             pred_q90=("pred_q90","mean"))
    )
    one["hour"] = one["timestamp"].dt.hour
    one["in_80"] = ((one["true"] >= one["pred_q10"]) & (one["true"] <= one["pred_q90"])).astype(int)

    return final_df, one
