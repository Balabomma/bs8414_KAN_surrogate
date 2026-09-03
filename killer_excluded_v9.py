"""Compute pooled valid/test R2 for models_v9 excluding the intrinsically-chaotic
Test_1_PE_PIR HRR1333 killer family (oracle-proven unpredictable, VARIANCE_RECORD
2026-07-08). Pooled R2 is recomputed on the concatenated masked points, not by
averaging per-sim R2. Saves a report to models_v9/outputs/killer_excluded_metrics.txt.
"""
import os
import argparse
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

from config import N_SENSORS
from data_loader import build_dataset, prepare_data_splits
from features_v6 import build_params_v6
from anchor_features import anchors_for
# v9 uses the v7 ambient-clamp model class — patch as evaluate_v9.py does.
import evaluate_v6
from model_v7 import KANAttentionLSTMv7
evaluate_v6.KANAttentionLSTMv6 = KANAttentionLSTMv7
from evaluate_v6 import load_ensemble_v6, predict_v6, inverse_scale

# Exclusion set: the documented chaotic killer family (Test_1_PE_PIR HRR1333).
def is_killer(m):
    return m["cladding"] == "Test_1_PE_PIR" and m["hrr"] == 1333


def pooled(actual, pred, mask, keep):
    a, p = [], []
    for j in keep:
        mk = mask[j].astype(bool)
        a.append(actual[j][mk].flatten())
        p.append(pred[j][mk].flatten())
    a = np.concatenate(a); p = np.concatenate(p)
    r2 = r2_score(a, p)
    rmse = float(np.sqrt(mean_squared_error(a, p)))
    hot = a > 100
    mape = float(np.mean(np.abs((a[hot] - p[hot]) / a[hot])) * 100) if hot.any() else float("nan")
    return r2, rmse, mape, a.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="models_v9")
    MODEL_DIR = ap.parse_args().model_dir
    models, weights, mean, scale, sensor_names, bank = load_ensemble_v6(MODEL_DIR)
    params, outputs, masks, meta, _ = build_dataset()
    _, _, _, _, split_info, time_array = prepare_data_splits(params, outputs, masks, meta)
    params_v6 = build_params_v6(params, meta, bank)

    lines = []
    lines.append("=" * 68)
    lines.append(f"  Killer-excluded pooled R2  ({MODEL_DIR})")
    lines.append("  Excluded family: Test_1_PE_PIR HRR1333 (LES-chaotic ignition spike;")
    lines.append("  oracle-proven unpredictable even with perfect anchors, 2026-07-08).")
    lines.append("=" * 68)

    for set_name in ("Valid", "Test"):
        idx = split_info[f"{set_name.lower()}_idx"]
        set_meta = [meta[i] for i in idx]
        anchors = anchors_for(params[idx], bank)
        pred = inverse_scale(predict_v6(models, weights, params_v6[idx], anchors, time_array),
                             mean, scale)
        actual = outputs[idx]
        mk = masks[idx]

        keep_all = list(range(len(idx)))
        killer_pos = [j for j, m in enumerate(set_meta) if is_killer(m)]
        keep_excl = [j for j in keep_all if j not in killer_pos]

        r2a, rmsea, mapea, na = pooled(actual, pred, mk, keep_all)
        r2e, rmsee, mapee, ne = pooled(actual, pred, mk, keep_excl)

        killer_names = [f"{set_meta[j]['cladding']} HRR{set_meta[j]['hrr']} {set_meta[j]['mesh']}"
                        for j in killer_pos]
        lines.append(f"\n[{set_name}]  {len(idx)} sims, killer excluded: {killer_names or 'none in bucket'}")
        lines.append(f"  pooled  (all {len(keep_all)}):   R2={r2a:.4f}  RMSE={rmsea:.1f}C  MAPE(T>100)={mapea:.1f}%")
        lines.append(f"  pooled  (excl killer, {len(keep_excl)}): R2={r2e:.4f}  RMSE={rmsee:.1f}C  MAPE(T>100)={mapee:.1f}%")

    report = "\n".join(lines)
    print(report)
    out = os.path.join(MODEL_DIR, "outputs", "killer_excluded_metrics.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(report + "\n")
    print(f"\n  Saved -> {out}")


if __name__ == "__main__":
    main()
