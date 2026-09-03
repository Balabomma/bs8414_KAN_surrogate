"""Training v5 — v4 (extended FDS inputs + anchors) + causal physics.

Adds to the v4 recipe:
  - model_v5.KANAttentionLSTMv5: causal driver channels q(t), Q(t) from the
    fire ramp boundary condition.
  - PhysicsLossV5: growth-phase monotonicity (T must not fall while the
    burner ramps up, t < 600 s), decay-phase constraint (T must not rise
    after burner shutdown, t > 1560 s), and an energy-consistency term
    (batch correlation between HRR input and predicted mean temperature).
  - Saves to models_v5/

Usage: python train_v5.py
"""
import os, time as time_mod, copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_squared_error

from config import N_SENSORS, N_TIMESTEPS, DEVICE, PROJECT_DIR
from data_loader import build_dataset, prepare_data_splits
from model import count_parameters, kan_regularization
from model_v5 import KANAttentionLSTMv5
from features_v4 import build_params_v4, N_V4_FEATURES
from anchor_features import build_bank, anchors_for
from train_v3 import (
    AugDatasetV3, EvalDataset, PhysicsLossV3, EMA,
    HIDDEN_SIZE, EMBEDDING_DIM, N_HEADS, DROPOUT, LEARNING_RATE, WEIGHT_DECAY,
    BATCH_SIZE, NUM_EPOCHS, PATIENCE, NUM_KNOTS, N_CANDIDATES, N_KEEP,
    LAMBDA_KAN_REG, pooled_r2,
)

N_CONTINUOUS = 2 + 13 + N_V4_FEATURES  # 34
MODEL_DIR = os.path.join(PROJECT_DIR, "models_v5")

LAMBDA_GROWTH = 0.05    # growth-phase monotonicity (t < 600 s)
LAMBDA_DECAY = 0.02     # decay-phase (t > 1560 s)
LAMBDA_ENERGY = 0.01    # HRR-vs-meanT batch correlation
GROWTH_END_IDX = 60     # 600 s
DECAY_START_IDX = 156   # 1560 s


class PhysicsLossV5(PhysicsLossV3):
    """v3 loss + causal growth/decay monotonicity + energy consistency."""

    def forward(self, pred, target, masks=None, hrr=None):
        base, _ = super().forward(pred, target, masks)

        # Growth phase: temperatures should not fall while the burner ramps up
        dg = pred[:, 1:GROWTH_END_IDX, :] - pred[:, :GROWTH_END_IDX - 1, :]
        growth = F.relu(-dg).pow(2).mean()

        # Decay phase: temperatures should not rise after burner shutdown
        dd = pred[:, DECAY_START_IDX + 1:, :] - pred[:, DECAY_START_IDX:-1, :]
        decay = F.relu(dd).pow(2).mean()

        # Energy consistency: higher HRR -> higher mean predicted temperature
        energy = pred.new_tensor(0.0)
        if hrr is not None and hrr.numel() > 2:
            mean_t = pred.mean(dim=(1, 2))
            h = hrr - hrr.mean()
            m = mean_t - mean_t.mean()
            denom = h.std() * m.std()
            if denom > 1e-6:
                energy = 1.0 - (h * m).mean() / denom

        total = (base + LAMBDA_GROWTH * growth + LAMBDA_DECAY * decay
                 + LAMBDA_ENERGY * energy)
        return total, {}


def make_model(device):
    return KANAttentionLSTMv5(
        n_timesteps=N_TIMESTEPS,
        n_sensors=N_SENSORS, hidden_size=HIDDEN_SIZE,
        embedding_dim=EMBEDDING_DIM, n_heads=N_HEADS,
        dropout=DROPOUT, num_knots=NUM_KNOTS, n_continuous=N_CONTINUOUS,
    ).to(device)


def train_epoch(model, loader, opt, criterion, device, ema):
    model.train()
    total, n = 0, 0
    for p, t, m, ta, a in loader:
        p, t, m, a = p.to(device), t.to(device), m.to(device), a.to(device)
        ta = ta[0].to(device)
        opt.zero_grad()
        pred, _ = model(p, ta, a)
        loss, _ = criterion(pred, t, m, hrr=p[:, 1])
        loss = loss + LAMBDA_KAN_REG * kan_regularization(model)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        ema.update(model)
        total += loss.item()
        n += 1
    return total / n


def validate(model, loader, criterion, device):
    model.eval()
    total, n = 0, 0
    with torch.no_grad():
        for p, t, m, ta, a in loader:
            p, t, m, a = p.to(device), t.to(device), m.to(device), a.to(device)
            ta = ta[0].to(device)
            total += criterion(model(p, ta, a)[0], t, m, hrr=p[:, 1])[0].item()
            n += 1
    return total / n


def member_preds_degC(model, params_np, anchors_np, time_array, scaler, device):
    model.eval()
    with torch.no_grad():
        p = torch.FloatTensor(params_np).to(device)
        a = torch.FloatTensor(anchors_np).to(device)
        ta = torch.FloatTensor(time_array).to(device)
        out = model(p, ta, a)[0].cpu().numpy()
    return (out.reshape(-1, N_SENSORS) * scaler.scale_ + scaler.mean_).reshape(out.shape)


def main():
    print("=" * 70)
    print("  BS8414 KAN-Attention-LSTM Surrogate — Training v5 (causal physics)")
    print("=" * 70)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.backends.cudnn.benchmark = True

    os.makedirs(MODEL_DIR, exist_ok=True)

    params, outputs, masks, meta, sensor_names = build_dataset()
    _, _, _, scaler, split_info, time_array = \
        prepare_data_splits(params, outputs, masks, meta)

    params_v4 = build_params_v4(params, meta)
    print(f"\n  Input vector: {params_v4.shape[1]} features")
    print(f"  Causal: q(t)/Q(t) channels; growth={LAMBDA_GROWTH} "
          f"decay={LAMBDA_DECAY} energy={LAMBDA_ENERGY}")

    np.save(os.path.join(MODEL_DIR, "output_scaler_mean.npy"), scaler.mean_)
    np.save(os.path.join(MODEL_DIR, "output_scaler_scale.npy"), scaler.scale_)
    np.save(os.path.join(MODEL_DIR, "sensor_names.npy"), sensor_names)

    train_idx = split_info["train_idx"]
    valid_idx = split_info["valid_idx"]
    test_idx = split_info["test_idx"]

    bank = build_bank(params, outputs, train_idx)
    np.savez(os.path.join(MODEL_DIR, "anchor_bank.npz"), **bank)

    train_anchors = anchors_for(params[train_idx], bank,
                                loo_bank_pos=np.arange(len(train_idx)))
    valid_anchors = anchors_for(params[valid_idx], bank)
    test_anchors = anchors_for(params[test_idx], bank)
    print(f"  Anchor LOO error (train): mean|err|="
          f"{np.abs(train_anchors - bank['anchor']).mean() * 1800:.0f}s")

    train_scaled = scaler.transform(
        outputs[train_idx].reshape(-1, N_SENSORS)
    ).reshape(-1, N_TIMESTEPS, N_SENSORS).astype(np.float32)
    valid_scaled = scaler.transform(
        outputs[valid_idx].reshape(-1, N_SENSORS)
    ).reshape(-1, N_TIMESTEPS, N_SENSORS).astype(np.float32)

    aug = AugDatasetV3(params_v4[train_idx], train_scaled, masks[train_idx],
                       time_array, train_anchors)
    train_loader = DataLoader(aug, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(
        EvalDataset(params_v4[valid_idx], valid_scaled, masks[valid_idx],
                    time_array, valid_anchors),
        batch_size=BATCH_SIZE)

    valid_actual = outputs[valid_idx]
    test_actual = outputs[test_idx]

    print(f"  Train: {len(train_idx)} real -> {len(aug)} augmented")

    init_temp = torch.FloatTensor(
        scaler.transform(np.full((1, N_SENSORS), 18.0, dtype=np.float32))[0]
    ).to(device)
    criterion = PhysicsLossV5(
        init_temp,
        torch.FloatTensor(scaler.mean_), torch.FloatTensor(scaler.scale_),
    ).to(device)

    print(f"\n--- Training {N_CANDIDATES} candidate members ---")
    members, val_losses, valid_pred_stack = [], [], []
    t0 = time_mod.time()

    for m_idx in range(N_CANDIDATES):
        seed = 42 + m_idx * 100
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        model = make_model(device)
        if m_idx == 0:
            print(f"  Params: {count_parameters(model):,}")

        opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=75, T_mult=2, eta_min=1e-5)
        ema = EMA(model, decay=0.999)
        eval_model = make_model(device)

        best_v, best_s, pat = float("inf"), None, 0
        for ep in range(1, NUM_EPOCHS + 1):
            train_epoch(model, train_loader, opt, criterion, device, ema)
            eval_model.load_state_dict(ema.apply_to(model))
            vl = validate(eval_model, valid_loader, criterion, device)
            sched.step()
            if vl < best_v:
                best_v = vl
                best_s = copy.deepcopy(eval_model.state_dict())
                pat = 0
            else:
                pat += 1
                if pat >= PATIENCE:
                    break

        model.load_state_dict(best_s)
        members.append(model)
        val_losses.append(best_v)
        vp = member_preds_degC(model, params_v4[valid_idx], valid_anchors,
                               time_array, scaler, device)
        valid_pred_stack.append(vp)
        vr2 = r2_score(valid_actual.flatten(), vp.flatten())
        print(f"  [M{m_idx+1:2d}] valid loss={best_v:.4f}  valid R2={vr2:.4f}  "
              f"({time_mod.time()-t0:.0f}s elapsed)")

    valid_pred_stack = np.stack(valid_pred_stack)

    torch.save({
        "model_states": [m.state_dict() for m in members],
        "n_models": len(members),
        "val_losses": val_losses,
    }, os.path.join(MODEL_DIR, "all_candidates.pt"))

    print(f"\n--- Greedy ensemble selection (max {N_KEEP} members, by valid R²) ---")
    selected, best_r2 = [], -np.inf
    while len(selected) < N_KEEP:
        best_cand, best_cand_r2 = None, best_r2
        for c in range(N_CANDIDATES):
            if c in selected:
                continue
            trial = selected + [c]
            r2 = pooled_r2(valid_pred_stack[trial],
                           [1.0 / val_losses[t] for t in trial], valid_actual)
            if r2 > best_cand_r2:
                best_cand, best_cand_r2 = c, r2
        if best_cand is None:
            break
        selected.append(best_cand)
        best_r2 = best_cand_r2
        print(f"  + M{best_cand+1}  -> valid R2={best_r2:.4f}")

    weights = np.array([1.0 / val_losses[i] for i in selected])
    weights = weights / weights.sum()
    ensemble = [members[i] for i in selected]
    print(f"  Selected: {[f'M{i+1}' for i in selected]}  "
          f"weights={[f'{w:.3f}' for w in weights]}")
    print(f"  Time: {time_mod.time()-t0:.0f}s")

    torch.save({
        "model_states": [m.state_dict() for m in ensemble],
        "n_models": len(ensemble),
        "ensemble_weights": weights.tolist(),
        "selected_candidates": [i + 1 for i in selected],
        "candidate_val_losses": val_losses,
        "n_continuous": N_CONTINUOUS,
    }, os.path.join(MODEL_DIR, "best_model.pt"))

    for name, sp, sa, an in [("Valid", params_v4[valid_idx], valid_actual, valid_anchors),
                             ("Test", params_v4[test_idx], test_actual, test_anchors)]:
        stack = np.stack([member_preds_degC(m, sp, an, time_array, scaler, device)
                          for m in ensemble])
        ens = np.tensordot(weights, stack, axes=(0, 0))
        r2 = r2_score(sa.flatten(), ens.flatten())
        rmse = np.sqrt(mean_squared_error(sa.flatten(), ens.flatten()))
        hot = sa.flatten() > 100
        mape = np.mean(np.abs((sa.flatten()[hot] - ens.flatten()[hot])
                              / sa.flatten()[hot])) * 100
        print(f"\n  {name}: R2={r2:.4f}  RMSE={rmse:.1f}C  MAPE(T>100)={mape:.1f}%")
        for k in range(len(sa)):
            print(f"    sim {k}: R2={r2_score(sa[k].flatten(), ens[k].flatten()):.3f}")

    print(f"\n  Saved -> {os.path.join(MODEL_DIR, 'best_model.pt')}")


if __name__ == "__main__":
    main()
