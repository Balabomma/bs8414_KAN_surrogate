"""Per-sensor and per-case test metrics for the Part1 sensor surrogate.

Does not re-score anything: imports `evaluate_part1`, reuses its own model
loading and `predict()` unaltered, and reports R2 and RMSE broken down by
individual thermocouple and by held-out case, on the identical predictions and
the identical mask the frozen contract uses.

    python dump_per_sensor.py --model-dir models_part1_bal_s42_r1 --out out.json
"""
import argparse
import io
import json
import sys

import numpy as np
import torch

import evaluate_part1 as E
from config_part1 import DEVICE
from data_loader_part1 import build_dataset, prepare_data_splits

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def r2_rmse(pred, true, m):
    """R2 against the channel mean, and RMSE, over masked points only."""
    if m.sum() == 0:
        return float("nan"), float("nan")
    p, t = pred[m], true[m]
    ss_res = float(((p - t) ** 2).sum())
    ss_tot = float(((t - t.mean()) ** 2).sum())
    rmse = float(np.sqrt(((p - t) ** 2).mean()))
    return (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"), rmse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    models, tc_scaler, hrr_scaler, ckpt = E.load_ensemble(a.model_dir, device)
    params, tc, hrr, mask_all, meta_all, sensor_names = build_dataset(verbose=False)
    datasets, _, _, info, _ = prepare_data_splits(
        params, tc, hrr, mask_all, meta_all, mode=ckpt.get("split_mode"), verbose=False)
    ds = datasets[a.split]

    tc_pred, _, mask = E.predict(models, ds, device, tc_scaler, hrr_scaler)
    tc_true = tc_scaler.inverse(ds.tc.numpy())
    chids = [m["chid"] for m in info["meta"][a.split]]

    # mask is (n_sims, n_steps) or (n_sims, n_steps, n_sensors)
    if mask.ndim == 2:
        M = np.repeat(mask[:, :, None], tc_true.shape[-1], axis=2)
    else:
        M = mask
    M = M.astype(bool)

    out = {"model_dir": a.model_dir, "split": a.split,
           "sensor_names": list(sensor_names), "per_sensor": {}, "per_case": {}}

    for j, nm in enumerate(sensor_names):
        r, e = r2_rmse(tc_pred[:, :, j], tc_true[:, :, j], M[:, :, j])
        out["per_sensor"][nm] = {"r2": r, "rmse": e}

    for i, chid in enumerate(chids):
        r, e = r2_rmse(tc_pred[i], tc_true[i], M[i])
        out["per_case"][chid] = {"r2": r, "rmse": e}

    out["pooled"] = {"r2": float(E.masked_r2(tc_pred, tc_true, mask)),
                     "rmse": float(E.masked_rmse(tc_pred, tc_true, mask))}
    r = out["pooled"]["r2"]

    json.dump(out, io.open(a.out, "w", encoding="utf-8"), indent=2)
    print("wrote %s" % a.out)
    print("pooled R2 %.4f  (contract %.4f)" % (r, E.masked_r2(tc_pred, tc_true, mask)))
    print("sensors: %d, cases: %d" % (len(sensor_names), len(chids)))


if __name__ == "__main__":
    main()
