"""Extended per-sensor test metrics for the Part1 sensor surrogate.

Same contract as `dump_per_sensor.py`, and deliberately built on it: imports
`evaluate_part1`, reuses its own model loading and `predict()` unaltered, and
computes every metric on the identical predictions and the identical mask the
frozen contract uses. Nothing is re-scored and no model is retrained.

It adds the metrics `dump_per_sensor.py` does not emit, so that Chapter 4 can
report the same lenses Chapter 5 reports for the field surrogate: mean absolute
error, mean bias error, the 95th-percentile absolute error, and the peak error,
which is the quantity 4.4.2 turns on.

    python dump_per_sensor_ext.py --model-dir models_part1_bal_s48_r1 --out out.json

Peak error is per case and per sensor: the peak of the masked prediction minus
the peak of the masked truth, averaged over held-out cases. Negative is
under-prediction of the peak, which is the unsafe direction.
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


def metrics(pred, true, m):
    """Every metric over masked points only. Shapes are (n_cases, n_steps)."""
    if m.sum() == 0:
        return {k: float("nan") for k in
                ("r2", "rmse", "mae", "mbe", "p95_abs_err", "n")}
    p, t = pred[m], true[m]
    d = p - t
    ss_res = float((d ** 2).sum())
    ss_tot = float(((t - t.mean()) ** 2).sum())
    return {
        "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "rmse": float(np.sqrt((d ** 2).mean())),
        "mae": float(np.abs(d).mean()),
        "mbe": float(d.mean()),
        "p95_abs_err": float(np.percentile(np.abs(d), 95)),
        "n": int(m.sum()),
    }


def peak_error(pred, true, m):
    """Mean over cases of (predicted peak - true peak), masked. Negative = under."""
    errs = []
    for i in range(pred.shape[0]):
        mi = m[i]
        if mi.sum() == 0:
            continue
        errs.append(float(pred[i][mi].max() - true[i][mi].max()))
    return float(np.mean(errs)) if errs else float("nan")


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

    if mask.ndim == 2:
        M = np.repeat(mask[:, :, None], tc_true.shape[-1], axis=2)
    else:
        M = mask
    M = M.astype(bool)

    out = {"model_dir": a.model_dir, "split": a.split,
           "sensor_names": list(sensor_names), "per_sensor": {}, "per_case": {}}

    for j, nm in enumerate(sensor_names):
        d = metrics(tc_pred[:, :, j], tc_true[:, :, j], M[:, :, j])
        d["peak_err"] = peak_error(tc_pred[:, :, j], tc_true[:, :, j], M[:, :, j])
        out["per_sensor"][nm] = d

    for i, chid in enumerate(chids):
        out["per_case"][chid] = metrics(tc_pred[i], tc_true[i], M[i])

    out["pooled"] = metrics(tc_pred, tc_true, M)
    # metrics() pools R2 against one global mean; the frozen contract uses a
    # per-channel mean and returns a lower number. Take the contract's.
    out["pooled"]["r2_global_mean"] = out["pooled"]["r2"]
    out["pooled"]["r2"] = float(E.masked_r2(tc_pred, tc_true, mask))
    out["pooled"]["peak_err"] = float(np.mean(
        [peak_error(tc_pred[:, :, j], tc_true[:, :, j], M[:, :, j])
         for j in range(len(sensor_names))]))

    json.dump(out, io.open(a.out, "w", encoding="utf-8"), indent=2)
    print("wrote %s" % a.out)
    print("pooled R2 %.4f  (contract %.4f)  RMSE %.2f  peak_err %+.2f"
          % (out["pooled"]["r2"], E.masked_r2(tc_pred, tc_true, mask),
             out["pooled"]["rmse"], out["pooled"]["peak_err"]))


if __name__ == "__main__":
    main()
