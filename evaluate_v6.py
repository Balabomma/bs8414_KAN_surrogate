"""Evaluate a saved v6 KAN ensemble — metrics, plots and CSVs.

Same outputs as evaluate_v5.py but for the v6 model (39-feature physics-causal
input vector, plume channel, anchor confidence).

Usage:
    python evaluate_v6.py --model-dir models_v6
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from config import (
    DEVICE, N_SENSORS, N_TIMESTEPS, T_END, KEY_THERMOCOUPLES,
)
from data_loader import build_dataset, prepare_data_splits
from features_v6 import build_params_v6, N_V6_FEATURES
from model_v6 import KANAttentionLSTMv6
from anchor_features import anchors_for
from train_v3 import HIDDEN_SIZE, EMBEDDING_DIM, N_HEADS, DROPOUT, NUM_KNOTS
from evaluate_kan import (
    plot_scatter, plot_timeseries, plot_per_sensor_bar, plot_error_hist,
    inverse_scale,
)

device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")

N_CONTINUOUS = 2 + 13 + N_V6_FEATURES


def load_ensemble_v6(model_dir):
    ckpt = torch.load(os.path.join(model_dir, "best_model.pt"),
                      map_location=device, weights_only=False)
    weights = np.array(ckpt["ensemble_weights"], dtype=np.float32)
    models = []
    for sd in ckpt["model_states"]:
        m = KANAttentionLSTMv6(
            n_timesteps=N_TIMESTEPS,
            n_sensors=N_SENSORS, hidden_size=HIDDEN_SIZE,
            embedding_dim=EMBEDDING_DIM, n_heads=N_HEADS,
            dropout=DROPOUT, num_knots=NUM_KNOTS,
            n_continuous=ckpt.get("n_continuous", N_CONTINUOUS),
        ).to(device)
        m.load_state_dict(sd)
        m.eval()
        models.append(m)
    mean = np.load(os.path.join(model_dir, "output_scaler_mean.npy"))
    scale = np.load(os.path.join(model_dir, "output_scaler_scale.npy"))
    names = list(np.load(os.path.join(model_dir, "sensor_names.npy"),
                         allow_pickle=True))
    bank = dict(np.load(os.path.join(model_dir, "anchor_bank.npz")))
    return models, weights, mean, scale, names, bank


def predict_v6(models, weights, params, anchors, time_array):
    p = torch.FloatTensor(params).to(device)
    a = torch.FloatTensor(anchors).to(device)
    ta = torch.FloatTensor(time_array).to(device)
    with torch.no_grad():
        out = sum(m(p, ta, a)[0].cpu().numpy() * w for m, w in zip(models, weights))
    return out


def evaluate(model_dir):
    output_dir = os.path.join(model_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  KAN v6 evaluation - {model_dir}  (split 70/15/15)")
    print(f"{'='*70}")

    models, weights, mean, scale, sensor_names, bank = load_ensemble_v6(model_dir)
    print(f"  Ensemble: {len(models)} members, weights={[f'{w:.3f}' for w in weights]}")

    params, outputs, masks, meta, _ = build_dataset()
    _, _, _, _, split_info, time_array = \
        prepare_data_splits(params, outputs, masks, meta)
    params_v6 = build_params_v6(params, meta, bank)

    time_s = np.linspace(0, T_END, N_TIMESTEPS)

    sets = {
        "Train": split_info["train_idx"],
        "Valid": split_info["valid_idx"],
        "Test": split_info["test_idx"],
    }

    all_metrics = {}
    test_pred = test_actual = test_mask = test_meta = None

    for set_name, idx in sets.items():
        if len(idx) == 0:
            continue
        set_meta = [meta[i] for i in idx]
        anchors = anchors_for(params[idx], bank)
        pred_scaled = predict_v6(models, weights, params_v6[idx], anchors, time_array)
        pred_degC = inverse_scale(pred_scaled, mean, scale)
        actual_degC = outputs[idx]
        mk = masks[idx]

        m = mk.astype(bool)
        a_flat = np.concatenate([actual_degC[i][m[i]].flatten() for i in range(len(idx))])
        p_flat = np.concatenate([pred_degC[i][m[i]].flatten() for i in range(len(idx))])
        r2 = r2_score(a_flat, p_flat)
        rmse = np.sqrt(mean_squared_error(a_flat, p_flat))
        mae = mean_absolute_error(a_flat, p_flat)
        hot = a_flat > 100
        mape = float(np.mean(np.abs((a_flat[hot] - p_flat[hot]) / a_flat[hot])) * 100) \
            if hot.any() else float("nan")
        all_metrics[set_name] = {"r2": r2, "rmse": rmse, "mae": mae,
                                 "mape": mape, "n": int(a_flat.size)}
        print(f"  {set_name}: R2={r2:.4f}  RMSE={rmse:.1f}C  MAE={mae:.1f}C  "
              f"MAPE(T>100)={mape:.1f}%")

        # per-sim R2 for valid/test transparency
        if set_name in ("Valid", "Test"):
            for j, i in enumerate(idx):
                sr2 = r2_score(actual_degC[j].flatten(), pred_degC[j].flatten())
                mm = meta[i]
                print(f"    {sr2:7.3f}  {mm['cladding']:<22} HRR{mm['hrr']} {mm['mesh']}")

        plot_scatter(actual_degC * mk[..., None], pred_degC * mk[..., None],
                     output_dir, set_name, "70/15/15")
        plot_error_hist(actual_degC * mk[..., None], pred_degC * mk[..., None],
                        output_dir, set_name)

        if set_name == "Test":
            test_pred, test_actual, test_mask, test_meta = \
                pred_degC, actual_degC, mk, set_meta
            for tc_name, tc_label in KEY_THERMOCOUPLES.items():
                if tc_name in sensor_names:
                    si = sensor_names.index(tc_name)
                    plot_timeseries(actual_degC, pred_degC, time_s, si, tc_label,
                                    set_meta, output_dir, set_name, mk)

    sensor_results = []
    if test_pred is not None:
        for s in range(N_SENSORS):
            m = test_mask.astype(bool)
            a = np.concatenate([test_actual[i, m[i], s]
                                for i in range(test_actual.shape[0])])
            p = np.concatenate([test_pred[i, m[i], s]
                                for i in range(test_pred.shape[0])])
            sensor_results.append({
                "sensor": sensor_names[s],
                "r2": r2_score(a, p),
                "rmse": float(np.sqrt(mean_squared_error(a, p))),
                "mae": float(mean_absolute_error(a, p)),
            })
        plot_per_sensor_bar(sensor_results, output_dir, "70/15/15")
        pd.DataFrame(sensor_results).to_csv(
            os.path.join(output_dir, "per_sensor_metrics.csv"), index=False)

    pd.DataFrame([{"Set": k, **v} for k, v in all_metrics.items()]).to_csv(
        os.path.join(output_dir, "overall_metrics.csv"), index=False)

    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(f"KAN v6 (physics-causal inputs) - {model_dir}\n")
        f.write("Split: 70/15/15 hash-based\n\n")
        for k, v in all_metrics.items():
            f.write(f"  {k}: R2={v['r2']:.4f}  RMSE={v['rmse']:.2f}C  "
                    f"MAE={v['mae']:.2f}C  MAPE(T>100)={v['mape']:.2f}%\n")
        if sensor_results:
            r2s = [s["r2"] for s in sensor_results]
            f.write(f"\n  Test per-sensor: avg R2={np.mean(r2s):.4f}\n")
            f.write(f"  Sensors R2>0.9:  {sum(1 for r in r2s if r > 0.9)}/{N_SENSORS}\n")
            f.write(f"  Sensors R2>0.8:  {sum(1 for r in r2s if r > 0.8)}/{N_SENSORS}\n")

    print(f"  Outputs -> {output_dir}")
    return all_metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="models_v6")
    args = p.parse_args()
    evaluate(args.model_dir)
