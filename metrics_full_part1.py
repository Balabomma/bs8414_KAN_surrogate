"""Additional descriptive metrics on the Part1 test split.

The frozen evaluation contract in `evaluate_part1.py` reports R2 and RMSE and is
deliberately not changed here: a candidate that needs different *scoring* is a
different experiment. This script does not re-score anything. It imports that
module, reuses its own model loading and prediction code unaltered, and reports
further descriptive statistics on the identical predictions, so that the thesis
can tabulate MAE, MAPE, MBE and NSE alongside the contract metrics.

    python metrics_full_part1.py --model-dir models_part1_184_r1 [--split test]

Every number printed is computed from the same `predict()` output the frozen
contract uses, on the same mask, in the same physical units (degC).
"""
import argparse
import io
import json
import os
import sys

import numpy as np
import torch

import evaluate_part1 as E
from config_part1 import DEVICE
from data_loader_part1 import build_dataset, prepare_data_splits

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# MAPE explodes as the denominator approaches zero, and a facade thermocouple
# sits at ambient for the first minutes of every test. Reporting it over all
# points would measure the ambient period rather than the fire. The threshold
# below is the one Table 4.1 of the thesis uses for the preliminary corpus.
MAPE_FLOOR_C = 100.0


def _flat(pred, true, mask):
    keep = mask.reshape(-1) > 0
    p = pred.reshape(-1, pred.shape[-1])[keep].reshape(-1)
    t = true.reshape(-1, true.shape[-1])[keep].reshape(-1)
    return p, t


def extra_metrics(pred, true, mask, floor=MAPE_FLOOR_C):
    p, t = _flat(pred, true, mask)
    err = p - t
    out = {
        "n_points": int(p.size),
        "mae": float(np.abs(err).mean()),
        "mbe": float(err.mean()),                       # signed bias
        "max_abs_err": float(np.abs(err).max()),
        "p95_abs_err": float(np.percentile(np.abs(err), 95)),
    }
    # Nash-Sutcliffe efficiency: 1 - SS_res/SS_tot with SS_tot taken about the
    # *pooled* mean of all observations. This is NOT the contract's R2, which
    # takes SS_tot about each channel's own mean and is therefore the stricter
    # figure. Both are reported because the difference between them measures how
    # much of the variance is between channels rather than within them.
    ss_res = float(((t - p) ** 2).sum())
    ss_tot = float(((t - t.mean()) ** 2).sum())
    out["nse"] = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    sel = t >= floor
    out["mape_pct"] = float((np.abs(err[sel] / t[sel])).mean() * 100.0) if sel.any() else float("nan")
    out["n_mape"] = int(sel.sum())
    out["mape_floor_c"] = floor
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--split", default="test", choices=["train", "valid", "test"])
    ap.add_argument("--split-mode", default=None, choices=("hash","system"))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    models, tc_scaler, hrr_scaler, ckpt = E.load_ensemble(a.model_dir, device)
    split_mode = a.split_mode or ckpt.get("split_mode")

    params, tc, hrr, mask_all, meta_all, sensor_names = build_dataset(verbose=False)
    datasets, _, _, info, _ = prepare_data_splits(
        params, tc, hrr, mask_all, meta_all, mode=split_mode, verbose=False)
    dataset = datasets[a.split]

    tc_pred, hrr_pred, mask = E.predict(models, dataset, device, tc_scaler, hrr_scaler)
    tc_true = tc_scaler.inverse(dataset.tc.numpy())

    print("model dir : %s" % a.model_dir)
    print("split     : %s  (mode %s)" % (a.split, split_mode))
    print()
    print("contract metrics (from evaluate_part1.py, unchanged)")
    print("   R2   %8.4f" % E.masked_r2(tc_pred, tc_true, mask))
    print("   RMSE %8.2f degC" % E.masked_rmse(tc_pred, tc_true, mask))
    print()
    m = extra_metrics(tc_pred, tc_true, mask)
    print("additional descriptive metrics (same predictions, same mask)")
    print("   MAE          %8.2f degC" % m["mae"])
    print("   MBE          %+8.2f degC   (signed; positive = over-prediction)" % m["mbe"])
    print("   MAPE         %8.2f %%      (over %d points with T >= %.0f degC)"
          % (m["mape_pct"], m["n_mape"], m["mape_floor_c"]))
    print("   NSE          %8.4f   (pooled-mean baseline; contract R2 uses per-channel means)" % m["nse"])
    print("   p95 |error|  %8.2f degC" % m["p95_abs_err"])
    print("   max |error|  %8.2f degC" % m["max_abs_err"])
    print("   n points     %8d" % m["n_points"])
    print("   n sims       %8d" % len(dataset))

    if a.out:
        json.dump(m, io.open(a.out, "w", encoding="utf-8"), indent=2)
        print("\nwrote %s" % a.out)


if __name__ == "__main__":
    main()
