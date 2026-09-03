"""Write per-case thermocouple time series for the Part1 test split.

Figure 4.1 of the thesis overlays the FDS target with each arm's prediction for
the two BR 135 regulatory thermocouples. The figure was originally built from
the preliminary corpus, whose evaluation persisted `ts_*.csv` files; the Part1
evaluation does not persist anything per case, so this script produces the same
files from the Part1 models.

It re-uses `evaluate_part1.py` unchanged for model loading and prediction, so
the series written here are exactly the ones the frozen contract scores.

    python dump_ts_part1.py --model-dir models_part1_kanbal_s48_r1 \
                            --pred-col KAN_Predicted_degC --out outputs_part1

One CSV per (case, thermocouple), columns: Time_s, FDS_Actual_degC, <pred-col>.
"""
import argparse
import io
import os
import sys

import numpy as np
import torch

import evaluate_part1 as E
from config_part1 import DEVICE, SENSOR_GROUPS, DT_DEVC
from data_loader_part1 import build_dataset, prepare_data_splits

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# The two BR 135 regulatory positions, by their BRE numbering.
KEY_TCS = ("1003", "1029")


def sensor_index(sensor_names, tc_id):
    """Column of the thermocouple whose name carries this BRE number."""
    for i, nm in enumerate(sensor_names):
        if tc_id in nm:
            return i
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--pred-col", required=True)
    ap.add_argument("--out", default="outputs_part1")
    ap.add_argument("--split", default="test")
    ap.add_argument("--cases", default=None,
                    help="comma-separated CHIDs; default = every case in the split")
    a = ap.parse_args()

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    models, tc_scaler, hrr_scaler, ckpt = E.load_ensemble(a.model_dir, device)
    params, tc, hrr, mask_all, meta_all, sensor_names = build_dataset(verbose=False)
    datasets, _, _, info, _ = prepare_data_splits(
        params, tc, hrr, mask_all, meta_all,
        mode=ckpt.get("split_mode"), verbose=False)

    ds = datasets[a.split]
    tc_pred, _, mask = E.predict(models, ds, device, tc_scaler, hrr_scaler)
    tc_true = tc_scaler.inverse(ds.tc.numpy())
    chids = [m["chid"] for m in info["meta"][a.split]]

    want = set(a.cases.split(",")) if a.cases else None
    os.makedirs(a.out, exist_ok=True)
    written = 0
    for i, chid in enumerate(chids):
        if want and chid not in want:
            continue
        n_steps = int(mask[i].sum()) if mask[i].ndim == 1 else int(mask[i].any(axis=-1).sum())
        for tc_id in KEY_TCS:
            j = sensor_index(sensor_names, tc_id)
            if j is None:
                continue
            path = os.path.join(a.out, "ts_%s_%s.csv" % (chid, tc_id))
            with io.open(path, "w", encoding="utf-8") as f:
                f.write("Time_s,FDS_Actual_degC,%s\n" % a.pred_col)
                for k in range(n_steps):
                    f.write("%.1f,%.4f,%.4f\n"
                            % (k * DT_DEVC, tc_true[i, k, j], tc_pred[i, k, j]))
            written += 1
    print("model dir : %s" % a.model_dir)
    print("split     : %s  (%d cases)" % (a.split, len(chids)))
    print("written   : %d files -> %s" % (written, a.out))
    if want:
        missing = want - set(chids)
        if missing:
            print("NOT in this split: %s" % ", ".join(sorted(missing)))
    else:
        print("cases     : %s" % ", ".join(chids[:6]) + (" ..." if len(chids) > 6 else ""))


if __name__ == "__main__":
    main()
