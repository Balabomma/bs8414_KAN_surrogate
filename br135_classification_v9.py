"""BR 135 external fire-spread classification — v9 surrogate vs FDS ground truth.

For each held-out test simulation (and each source: FDS actual, v9 prediction):
  t_s   = first time the mean External LV1 temperature exceeds ambient + 200 degC
          (BS 8414 start-of-test condition, level-1 thermocouples above the chamber)
  FAIL  = any External LV2 thermocouple >= ambient + 600 degC sustained >= 30 s
          (3 consecutive 10-s samples) within 900 s (15 min) of t_s
Reports the per-sim classification and whether surrogate matches FDS.

Usage: python br135_classification_v9.py
"""
import numpy as np

from config import T_END, N_TIMESTEPS
from data_loader import build_dataset, prepare_data_splits
from features_v6 import build_params_v6
from anchor_features import anchors_for
import evaluate_v6
from model_v7 import KANAttentionLSTMv7

evaluate_v6.KANAttentionLSTMv6 = KANAttentionLSTMv7

T_AMB = 18.0
DT = T_END / (N_TIMESTEPS - 1)          # 10 s
START_RISE = 200.0                      # t_s condition on Level 1
FAIL_RISE = 600.0                       # BR 135 external spread threshold
SUSTAIN_SECONDS = [30, 90, 270]         # sustained-exceedance durations (seconds)
WINDOW = int(900 / DT)                  # 15 min after t_s


def classify(hist, lv1_idx, lv2_idx, sustain_samples):
    """hist: (T, S) degC. sustain_samples = required consecutive 10-s samples.
    Returns (t_s [s] or None, 'FAIL'/'PASS', t_fail [s] or None)."""
    lv1_mean = hist[:, lv1_idx].mean(axis=1)
    ts_cand = np.where(lv1_mean >= T_AMB + START_RISE)[0]
    if len(ts_cand) == 0:
        return None, "PASS", None          # never starts -> no spread
    ts = int(ts_cand[0])
    end = min(ts + WINDOW, hist.shape[0])
    over = hist[ts:end, lv2_idx] >= T_AMB + FAIL_RISE   # (W, n_lv2)
    for s in range(over.shape[1]):
        run = 0
        for t in range(over.shape[0]):
            run = run + 1 if over[t, s] else 0
            if run >= sustain_samples:
                return ts * DT, "FAIL", (ts + t) * DT
    return ts * DT, "PASS", None


def main():
    models, weights, mean, scale, sensor_names, bank = \
        evaluate_v6.load_ensemble_v6("models_v9")
    params, outputs, masks, meta, _ = build_dataset()
    _, _, _, _, split_info, time_array = prepare_data_splits(params, outputs, masks, meta)
    params_v6 = build_params_v6(params, meta, bank)

    lv1_idx = [i for i, n in enumerate(sensor_names) if "External_LV1" in n]
    lv2_idx = [i for i, n in enumerate(sensor_names) if "External_LV2" in n]
    print(f"LV1 sensors: {len(lv1_idx)}, LV2 sensors: {len(lv2_idx)}")

    idx = split_info["test_idx"]
    anchors = anchors_for(params[idx], bank)
    pred = evaluate_v6.inverse_scale(
        evaluate_v6.predict_v6(models, weights, params_v6[idx], anchors, time_array),
        mean, scale)

    summary = []
    for sustain_s in SUSTAIN_SECONDS:
        sustain_samples = int(round(sustain_s / DT))
        print(f"\n{'='*80}")
        print(f"  BR 135 external-spread classification -- sustain >= {sustain_s} s "
              f"({sustain_samples} x {DT:.0f}-s samples), 600 degC rise on External LV2")
        print(f"{'='*80}")
        print(f"{'sim':<40} {'FDS':>18} {'v9 surrogate':>18} match")
        agree = 0
        for j, i in enumerate(idx):
            m = meta[i]
            label = f"{m['cladding']} H{m['hrr']} {m['mesh']}"
            fds_ts, fds_cls, fds_tf = classify(outputs[i], lv1_idx, lv2_idx, sustain_samples)
            sur_ts, sur_cls, sur_tf = classify(pred[j], lv1_idx, lv2_idx, sustain_samples)
            ok = fds_cls == sur_cls
            agree += ok
            ffs = f"{fds_cls}" + (f" @{fds_tf:.0f}s" if fds_tf else "")
            sfs = f"{sur_cls}" + (f" @{sur_tf:.0f}s" if sur_tf else "")
            print(f"{label:<40} {ffs:>18} {sfs:>18} {'YES' if ok else 'NO'}")
        print(f"\n  Agreement (sustain {sustain_s} s): {agree}/{len(idx)}")
        summary.append((sustain_s, agree, len(idx)))

    print(f"\n{'='*80}")
    print("  Summary -- surrogate vs FDS classification agreement by sustain duration")
    print(f"{'='*80}")
    for sustain_s, agree, n in summary:
        print(f"    sustain >= {sustain_s:>4} s : {agree}/{n} sims agree")


if __name__ == "__main__":
    main()
