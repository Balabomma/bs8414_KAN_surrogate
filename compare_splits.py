"""Compare 70/15/15 vs 80/10/10 KAN ensembles on the held-out test set.

Both models are evaluated on the same 5 simulations (bucket >= 0.9), which
are held out from training under both split schemes. Each model uses its
own output scaler for denormalization.
"""
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_squared_error

from config import (
    MODEL_DIR, N_SENSORS, N_TIMESTEPS, DEVICE,
    TRAIN_RATIO, VALID_RATIO, TEST_RATIO,
)
from data_loader import build_dataset, prepare_data_splits, FDSSurrogateDataset
from model import KANAttentionLSTM
from train import HIDDEN_SIZE, EMBEDDING_DIM, N_HEADS, DROPOUT, NUM_KNOTS

BACKUP_DIR = os.path.join(os.path.dirname(MODEL_DIR), "models_70_15_15_original_jun04")
device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print(f"Current split ratios: train={TRAIN_RATIO} valid={VALID_RATIO} test={TEST_RATIO}\n")


def load_ensemble(model_dir):
    ckpt = torch.load(os.path.join(model_dir, "best_model.pt"),
                      map_location=device, weights_only=False)
    weights = np.array(ckpt["ensemble_weights"])
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
    return models, weights, scaler_mean, scaler_scale


def predict(models, weights, params, time_array, scaler_mean, scaler_scale):
    """Run ensemble inference on (n,16) params; return denormalized (n,T,S)."""
    pt = torch.FloatTensor(params).to(device)
    ta = torch.FloatTensor(time_array).to(device)
    with torch.no_grad():
        out = sum(m(pt, ta)[0].cpu().numpy() * w for m, w in zip(models, weights))
    return (out.reshape(-1, N_SENSORS) * scaler_scale + scaler_mean).reshape(out.shape)


def metrics(pred, actual, mask):
    """Compute R2, RMSE, MAPE(T>100C), per-sensor R2 using mask."""
    m = mask.astype(bool)
    flat_pred = []
    flat_actual = []
    sensor_r2 = []
    for s in range(N_SENSORS):
        p_s = pred[:, :, s][m]
        a_s = actual[:, :, s][m]
        flat_pred.append(p_s)
        flat_actual.append(a_s)
        sensor_r2.append(r2_score(a_s, p_s))
    flat_pred = np.concatenate(flat_pred)
    flat_actual = np.concatenate(flat_actual)
    r2 = r2_score(flat_actual, flat_pred)
    rmse = np.sqrt(mean_squared_error(flat_actual, flat_pred))
    hot = flat_actual > 100
    mape = np.mean(np.abs((flat_actual[hot] - flat_pred[hot]) / flat_actual[hot])) * 100
    return r2, rmse, mape, np.array(sensor_r2)


# ── Build dataset and identify test set under current 80/10/10 ──────
params, outputs, masks, meta, sensor_names = build_dataset()
_, _, _, _, split_info, time_array = prepare_data_splits(params, outputs, masks, meta)
test_idx = split_info["test_idx"]
test_meta = split_info["test_meta"]
test_params = params[test_idx]
test_outputs = outputs[test_idx]       # raw degC, padded
test_masks = masks[test_idx]

print(f"\nHeld-out test sims (n={len(test_idx)}):")
for m in test_meta:
    print(f"  - {m['cladding']:<22} HRR{m['hrr']:<5} {m['mesh']}")

# ── Load both ensembles ─────────────────────────────────────────────
print("\nLoading 80/10/10 ensemble (current models/)...")
models_new, w_new, mean_new, scale_new = load_ensemble(MODEL_DIR)
print(f"  Weights: {[f'{w:.3f}' for w in w_new]}")

print("Loading 70/15/15 ensemble (backup)...")
models_old, w_old, mean_old, scale_old = load_ensemble(BACKUP_DIR)
print(f"  Weights: {[f'{w:.3f}' for w in w_old]}")

# ── Predict ─────────────────────────────────────────────────────────
pred_new = predict(models_new, w_new, test_params, time_array, mean_new, scale_new)
pred_old = predict(models_old, w_old, test_params, time_array, mean_old, scale_old)

# ── Metrics ─────────────────────────────────────────────────────────
r2_n, rmse_n, mape_n, sr2_n = metrics(pred_new, test_outputs, test_masks)
r2_o, rmse_o, mape_o, sr2_o = metrics(pred_old, test_outputs, test_masks)

print("\n" + "=" * 70)
print("  HEAD-TO-HEAD ON HELD-OUT TEST SET")
print("=" * 70)
print(f"  {'Metric':<25} {'70/15/15 (old)':>16} {'80/10/10 (new)':>16} {'delta':>10}")
print(f"  {'-'*25} {'-'*16} {'-'*16} {'-'*10}")
print(f"  {'Overall R2':<25} {r2_o:>16.4f} {r2_n:>16.4f} {r2_n - r2_o:>+10.4f}")
print(f"  {'Overall RMSE (C)':<25} {rmse_o:>16.2f} {rmse_n:>16.2f} {rmse_n - rmse_o:>+10.2f}")
print(f"  {'MAPE (T>100C) %':<25} {mape_o:>16.2f} {mape_n:>16.2f} {mape_n - mape_o:>+10.2f}")
print(f"  {'Avg per-sensor R2':<25} {sr2_o.mean():>16.4f} {sr2_n.mean():>16.4f} {sr2_n.mean() - sr2_o.mean():>+10.4f}")
print(f"  {'Sensors with R2 > 0.9':<25} {int((sr2_o > 0.9).sum()):>16d} {int((sr2_n > 0.9).sum()):>16d} {int((sr2_n > 0.9).sum() - (sr2_o > 0.9).sum()):>+10d}")
print(f"  {'Sensors with R2 > 0.8':<25} {int((sr2_o > 0.8).sum()):>16d} {int((sr2_n > 0.8).sum()):>16d} {int((sr2_n > 0.8).sum() - (sr2_o > 0.8).sum()):>+10d}")

# Key thermocouples (from config.KEY_THERMOCOUPLES)
key_tcs = {
    "External_LV2_main02(1029)": "TC 1029 - External LV2 Main",
    "External_LV1_main02(1003)": "TC 1003 - External LV1 Main",
}
print("\n  Key thermocouples:")
print(f"  {'Sensor':<45} {'70/15/15':>10} {'80/10/10':>10} {'delta':>8}")
for tc_name, label in key_tcs.items():
    if tc_name in sensor_names:
        idx = list(sensor_names).index(tc_name)
        print(f"  {label:<45} {sr2_o[idx]:>10.4f} {sr2_n[idx]:>10.4f} {sr2_n[idx] - sr2_o[idx]:>+8.4f}")

# Per-sim test R2
print("\n  Per-simulation overall R2 (across all 24 sensors × valid timesteps):")
print(f"  {'Sim':<55} {'70/15/15':>10} {'80/10/10':>10} {'delta':>8}")
for i, m in enumerate(test_meta):
    mk = test_masks[i].astype(bool)
    a = test_outputs[i][mk]
    po = pred_old[i][mk]
    pn = pred_new[i][mk]
    r2o = r2_score(a.flatten(), po.flatten())
    r2n = r2_score(a.flatten(), pn.flatten())
    chid = m["chid"][:53]
    print(f"  {chid:<55} {r2o:>10.4f} {r2n:>10.4f} {r2n - r2o:>+8.4f}")

print("=" * 70)

# Save numbers for later reference
out_path = os.path.join(os.path.dirname(MODEL_DIR), "split_comparison.txt")
with open(out_path, "w") as f:
    f.write(f"70/15/15 vs 80/10/10 KAN ensemble — held-out test set (n={len(test_idx)})\n")
    f.write(f"Test sims: {[m['chid'] for m in test_meta]}\n\n")
    f.write(f"Overall R2:        old={r2_o:.4f}  new={r2_n:.4f}  delta={r2_n-r2_o:+.4f}\n")
    f.write(f"Overall RMSE C:    old={rmse_o:.2f}  new={rmse_n:.2f}  delta={rmse_n-rmse_o:+.2f}\n")
    f.write(f"MAPE (T>100) %:    old={mape_o:.2f}  new={mape_n:.2f}  delta={mape_n-mape_o:+.2f}\n")
    f.write(f"Avg sensor R2:     old={sr2_o.mean():.4f}  new={sr2_n.mean():.4f}\n")
    f.write(f"Sensors R2>0.9:    old={(sr2_o>0.9).sum()}  new={(sr2_n>0.9).sum()}\n")
    f.write(f"Sensors R2>0.8:    old={(sr2_o>0.8).sum()}  new={(sr2_n>0.8).sum()}\n")
print(f"\n  Saved summary -> {out_path}")
