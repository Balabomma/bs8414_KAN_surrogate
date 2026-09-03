"""Evaluate a saved KAN-Attention-LSTM ensemble and write plots + CSVs.

Usage:
    python evaluate_kan.py --model-dir models                  --train-ratio 0.70 --valid-ratio 0.15
    python evaluate_kan.py --model-dir models_70_15_15         --train-ratio 0.70 --valid-ratio 0.15
    python evaluate_kan.py --model-dir models_70_15_15_original_jun04 --train-ratio 0.70 --valid-ratio 0.15
    python evaluate_kan.py --model-dir models_80_10_10         --train-ratio 0.80 --valid-ratio 0.10

Outputs land in <model-dir>/outputs/.
"""
import os
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from config import (
    DEVICE, N_SENSORS, N_TIMESTEPS, T_END,
    CLADDING_SYSTEMS, HRR_LEVELS, MESH_SIZES,
    KEY_THERMOCOUPLES,
)
from data_loader import build_dataset, assign_split, FDSSurrogateDataset
from model import KANAttentionLSTM
from train import HIDDEN_SIZE, EMBEDDING_DIM, N_HEADS, DROPOUT, NUM_KNOTS

matplotlib.rcParams.update({
    "font.family": "Arial", "font.size": 11, "axes.linewidth": 1.2,
    "axes.labelsize": 13, "axes.titlesize": 13, "axes.labelweight": "bold",
    "xtick.major.size": 6, "xtick.major.width": 1.2,
    "xtick.minor.size": 3, "xtick.minor.width": 0.8,
    "xtick.direction": "in", "xtick.top": True,
    "ytick.major.size": 6, "ytick.major.width": 1.2,
    "ytick.minor.size": 3, "ytick.minor.width": 0.8,
    "ytick.direction": "in", "ytick.right": True,
    "legend.frameon": True, "legend.edgecolor": "black", "legend.framealpha": 1.0,
    "legend.fontsize": 10, "figure.dpi": 200, "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")


def load_ensemble(model_dir):
    ckpt = torch.load(os.path.join(model_dir, "best_model.pt"),
                      map_location=device, weights_only=False)
    weights = np.array(ckpt["ensemble_weights"], dtype=np.float32)
    models = []
    for sd in ckpt["model_states"]:
        m = KANAttentionLSTM(
            n_sensors=N_SENSORS, hidden_size=HIDDEN_SIZE,
            embedding_dim=EMBEDDING_DIM, n_heads=N_HEADS,
            dropout=DROPOUT, num_knots=NUM_KNOTS,
        ).to(device)
        m.load_state_dict(sd)
        m.eval()
        models.append(m)
    scaler_mean = np.load(os.path.join(model_dir, "output_scaler_mean.npy"))
    scaler_scale = np.load(os.path.join(model_dir, "output_scaler_scale.npy"))
    sensor_names = list(np.load(os.path.join(model_dir, "sensor_names.npy"), allow_pickle=True))
    return models, weights, scaler_mean, scaler_scale, sensor_names


def predict(models, weights, params, time_array):
    pt = torch.FloatTensor(params).to(device)
    ta = torch.FloatTensor(time_array).to(device)
    with torch.no_grad():
        out = sum(m(pt, ta)[0].cpu().numpy() * w for m, w in zip(models, weights))
    return out


def inverse_scale(scaled, mean, scale):
    shape = scaled.shape
    return (scaled.reshape(-1, N_SENSORS) * scale + mean).reshape(shape)


def split_indices(meta, train_ratio, valid_ratio):
    tr, va, te = [], [], []
    for i, m in enumerate(meta):
        s = assign_split(m["chid"], train_ratio=train_ratio, valid_ratio=valid_ratio)
        if s == "train": tr.append(i)
        elif s == "valid": va.append(i)
        else: te.append(i)
    return np.array(tr), np.array(va), np.array(te)


def plot_scatter(actual, predicted, output_dir, set_label, ratio_label):
    a, p = actual.flatten(), predicted.flatten()
    r2 = r2_score(a, p)
    rmse = np.sqrt(mean_squared_error(a, p))
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(a, p, alpha=0.15, s=3, c="#1f77b4", edgecolors="none", rasterized=True)
    lims = [min(a.min(), p.min()) - 20, max(a.max(), p.max()) + 20]
    ax.plot(lims, lims, "k-", linewidth=1.5, label="y = x")
    x_line = np.linspace(lims[0], lims[1], 100)
    ax.fill_between(x_line, x_line * 0.9, x_line * 1.1, alpha=0.08, color="gray", label="+/-10%")
    ax.set_xlabel("FDS Temperature (°C)")
    ax.set_ylabel("KAN-Predicted Temperature (°C)")
    ax.set_title(f"KAN Ensemble - {set_label} ({ratio_label})")
    ax.set_xlim(lims); ax.set_ylim(lims); ax.set_aspect("equal")
    props = dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="black")
    ax.text(0.05, 0.95, f"R² = {r2:.4f}\nRMSE = {rmse:.1f} °C\nN = {len(a):,}",
            transform=ax.transAxes, fontsize=11, va="top", bbox=props)
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.legend(loc="lower right", fancybox=False); ax.grid(False)
    plt.tight_layout()
    fname = os.path.join(output_dir, f"scatter_{set_label}.png")
    plt.savefig(fname); plt.close()
    return fname


def plot_timeseries(actual, predicted, time_s, sensor_idx, sensor_label,
                    sims_meta, output_dir, set_label, mask):
    paths = []
    for i in range(actual.shape[0]):
        meta = sims_meta[i]
        chid_short = f"{meta['cladding']}_HRR{meta['hrr']}_{meta['mesh']}"
        m = mask[i].astype(bool)
        t = time_s[m]
        a = actual[i, m, sensor_idx]
        p = predicted[i, m, sensor_idx]
        fig, ax = plt.subplots(figsize=(8, 5.2))
        ax.plot(t, a, color="#1f77b4", linewidth=2.0, label="FDS Simulation")
        ax.plot(t, p, color="#d62728", linewidth=2.0, linestyle="--",
                label="KAN-Attention-LSTM (Ensemble)")
        err = np.abs(a - p)
        ax.fill_between(t, p - err, p + err, alpha=0.1, color="#d62728")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Temperature (°C)")
        hrr_mw = meta["hrr"] * 1.5 / 1000
        ax.set_title(f"{sensor_label}\n{meta['cladding']} | HRRPUA={meta['hrr']} kW/m² ({hrr_mw:.2f} MW) | Mesh={meta['mesh']}")
        ax.set_xlim(0, T_END)
        ax.xaxis.set_major_locator(MultipleLocator(300))
        ax.xaxis.set_minor_locator(MultipleLocator(60))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))
        r2 = r2_score(a, p); rmse = np.sqrt(mean_squared_error(a, p))
        props = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black")
        ax.text(0.97, 0.05, f"R²={r2:.3f}\nRMSE={rmse:.1f}°C",
                transform=ax.transAxes, fontsize=10, ha="right", bbox=props)
        ax.legend(loc="best", fancybox=False); ax.grid(False)
        plt.tight_layout()
        safe = sensor_label.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        fname = os.path.join(output_dir, f"ts_{safe}_{set_label}_{chid_short}.png")
        plt.savefig(fname); plt.close()
        pd.DataFrame({
            "Time_s": t, "FDS_Actual_degC": a,
            "KAN_Predicted_degC": p, "Abs_Error_degC": err,
        }).to_csv(fname.replace(".png", ".csv"), index=False)
        paths.append(fname)
    return paths


def plot_per_sensor_bar(sensor_results, output_dir, ratio_label):
    fig, ax = plt.subplots(figsize=(11, 5))
    names = [s["sensor"] for s in sensor_results]
    r2s = np.array([s["r2"] for s in sensor_results])
    colors = ["#2ca02c" if r > 0.9 else ("#ff7f0e" if r > 0.8 else "#d62728") for r in r2s]
    ax.bar(range(len(names)), r2s, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0.9, color="#2ca02c", linestyle="--", linewidth=1, label="R² = 0.9")
    ax.axhline(0.8, color="#ff7f0e", linestyle="--", linewidth=1, label="R² = 0.8")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("Test R²"); ax.set_ylim(min(0, r2s.min() - 0.05), 1.05)
    ax.set_title(f"KAN per-sensor test R² ({ratio_label})  -  avg={r2s.mean():.3f}")
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.legend(loc="lower right")
    plt.tight_layout()
    fname = os.path.join(output_dir, "per_sensor_R2.png")
    plt.savefig(fname); plt.close()
    return fname


def plot_error_hist(actual, predicted, output_dir, set_label):
    err = (predicted - actual).flatten()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(err, bins=60, color="#1f77b4", edgecolor="black", linewidth=0.4)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Prediction error (°C)  [pred - FDS]")
    ax.set_ylabel("Count")
    ax.set_title(f"KAN error distribution - {set_label}")
    me, sd = err.mean(), err.std()
    p95 = np.percentile(np.abs(err), 95)
    props = dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="black")
    ax.text(0.97, 0.95, f"mean={me:.1f}°C\nstd={sd:.1f}°C\n|err| P95={p95:.1f}°C",
            transform=ax.transAxes, fontsize=10, ha="right", va="top", bbox=props)
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    plt.tight_layout()
    fname = os.path.join(output_dir, f"error_hist_{set_label}.png")
    plt.savefig(fname); plt.close()
    return fname


def evaluate(model_dir, train_ratio, valid_ratio):
    ratio_label = f"{int(round(train_ratio*100))}/{int(round(valid_ratio*100))}/{int(round((1-train_ratio-valid_ratio)*100))}"
    output_dir = os.path.join(model_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  KAN evaluation - {model_dir}  (split {ratio_label})")
    print(f"{'='*70}")

    models, weights, mean, scale, sensor_names = load_ensemble(model_dir)
    print(f"  Ensemble: {len(models)} members, weights={[f'{w:.3f}' for w in weights]}")

    params, outputs, masks, meta, _ = build_dataset()
    train_idx, valid_idx, test_idx = split_indices(meta, train_ratio, valid_ratio)
    print(f"  Split: train={len(train_idx)}  valid={len(valid_idx)}  test={len(test_idx)}")

    time_array = np.linspace(0, 1, N_TIMESTEPS).astype(np.float32)
    time_s = np.linspace(0, T_END, N_TIMESTEPS)

    sets = {
        "Train": (train_idx, [meta[i] for i in train_idx]),
        "Valid": (valid_idx, [meta[i] for i in valid_idx]),
        "Test":  (test_idx,  [meta[i] for i in test_idx]),
    }

    all_metrics = {}
    test_pred = test_actual = test_mask = test_meta = None

    for set_name, (idx, set_meta) in sets.items():
        if len(idx) == 0:
            print(f"  {set_name}: empty - skipped")
            continue
        sp = params[idx]
        # outputs[idx] are RAW degC. Models output SCALED, so we need scaled targets only for the model output;
        # we just compare denormalised pred vs raw actual.
        pred_scaled = predict(models, weights, sp, time_array)
        pred_degC = inverse_scale(pred_scaled, mean, scale)
        actual_degC = outputs[idx]
        mk = masks[idx]

        # Apply mask for metrics (skip padded steps)
        m = mk.astype(bool)
        a_flat = np.concatenate([actual_degC[i][m[i]].flatten() for i in range(len(idx))])
        p_flat = np.concatenate([pred_degC[i][m[i]].flatten() for i in range(len(idx))])
        r2 = r2_score(a_flat, p_flat)
        rmse = np.sqrt(mean_squared_error(a_flat, p_flat))
        mae = mean_absolute_error(a_flat, p_flat)
        hot = a_flat > 100
        mape = float(np.mean(np.abs((a_flat[hot] - p_flat[hot]) / a_flat[hot])) * 100) if hot.any() else float("nan")
        all_metrics[set_name] = {"r2": r2, "rmse": rmse, "mae": mae, "mape": mape, "n": int(hot.size)}
        print(f"  {set_name}: R2={r2:.4f}  RMSE={rmse:.1f}C  MAE={mae:.1f}C  MAPE(T>100)={mape:.1f}%")

        # Scatter and error hist
        plot_scatter(actual_degC * mk[..., None], pred_degC * mk[..., None],
                     output_dir, set_name, ratio_label)
        plot_error_hist(actual_degC * mk[..., None], pred_degC * mk[..., None],
                        output_dir, set_name)

        # Key thermocouple time series for Test set only (keep file count manageable)
        if set_name == "Test":
            test_pred, test_actual, test_mask, test_meta = pred_degC, actual_degC, mk, set_meta
            for tc_name, tc_label in KEY_THERMOCOUPLES.items():
                if tc_name in sensor_names:
                    si = sensor_names.index(tc_name)
                    plot_timeseries(actual_degC, pred_degC, time_s, si, tc_label,
                                    set_meta, output_dir, set_name, mk)

    # Per-sensor (test)
    sensor_results = []
    if test_pred is not None:
        for s in range(N_SENSORS):
            m = test_mask.astype(bool)
            a = np.concatenate([test_actual[i, m[i], s] for i in range(test_actual.shape[0])])
            p = np.concatenate([test_pred[i, m[i], s] for i in range(test_pred.shape[0])])
            sensor_results.append({
                "sensor": sensor_names[s],
                "r2": r2_score(a, p),
                "rmse": float(np.sqrt(mean_squared_error(a, p))),
                "mae": float(mean_absolute_error(a, p)),
            })
        plot_per_sensor_bar(sensor_results, output_dir, ratio_label)
        pd.DataFrame(sensor_results).to_csv(
            os.path.join(output_dir, "per_sensor_metrics.csv"), index=False)

    # Overall metrics CSV
    pd.DataFrame([{"Set": k, **v} for k, v in all_metrics.items()]).to_csv(
        os.path.join(output_dir, "overall_metrics.csv"), index=False)

    # Summary text
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(f"KAN-Attention-LSTM evaluation - {model_dir}\n")
        f.write(f"Split: {ratio_label}  (train={len(train_idx)} valid={len(valid_idx)} test={len(test_idx)})\n")
        f.write(f"Ensemble weights: {weights.tolist()}\n\n")
        for k, v in all_metrics.items():
            f.write(f"  {k}: R2={v['r2']:.4f}  RMSE={v['rmse']:.2f}C  MAE={v['mae']:.2f}C  MAPE(T>100)={v['mape']:.2f}%\n")
        if sensor_results:
            n_above_09 = sum(1 for s in sensor_results if s["r2"] > 0.9)
            n_above_08 = sum(1 for s in sensor_results if s["r2"] > 0.8)
            f.write(f"\n  Test per-sensor: avg R2={np.mean([s['r2'] for s in sensor_results]):.4f}\n")
            f.write(f"  Sensors R2>0.9:  {n_above_09}/{N_SENSORS}\n")
            f.write(f"  Sensors R2>0.8:  {n_above_08}/{N_SENSORS}\n")

    print(f"  Outputs -> {output_dir}")
    return all_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", required=True)
    p.add_argument("--train-ratio", type=float, required=True)
    p.add_argument("--valid-ratio", type=float, required=True)
    args = p.parse_args()
    evaluate(args.model_dir, args.train_ratio, args.valid_ratio)


if __name__ == "__main__":
    main()
