"""Training v8 — v7 (ambient clamp) + robust tail-downweighted data loss.

Parent: train_v7 (models_v7). One variable changed: the data term of the loss.

Motivation (measured, not assumed):
  - The oracle experiment (perfect anchors -> 0 R2 gain; killer sim stays R2~0.37)
    refuted the anchor-predictor hypothesis. The R2 ceiling is 1-2 intrinsically
    chaotic sims (Test_1_PE_PIR HRR1333 family): dropping that one sim lifts valid
    R2 0.806->0.860 and test 0.892->0.913.
  - Scaled-residual distribution on train is heavy-tailed: p50=0.10, p90=0.39,
    p99=1.23, max=9.2. The peak-weighted MSE gives those unfittable spike points
    BOTH a huge squared error AND a large peak weight, so they dominate the
    gradient and starve the 61 fittable sims.

Change: multiply the per-point data term by a detached adaptive tail-weight
  tw = 1 / (1 + (|pred-target| / TAU)^2),  TAU = 1.0 (scaled units)
so points with persistently large residual (the chaotic spikes) shed gradient
while fittable points (including legitimate 900 degC peaks, |resid| small once
learned) keep full L2 pressure. Chosen over Huber/log-cosh precisely because those
cap gradients on ALL large errors, including real high-temperature peaks the model
should fit; the tail-weight only releases the genuinely-unfittable tail.

All physics terms (init, smooth, rel, growth, decay, energy), the ambient clamp,
features, split, and eval contract are identical to v7. Saves to models_v8/.

Usage: python train_v8.py
"""
import os, time as time_mod, copy
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score, mean_squared_error

from config import N_SENSORS, N_TIMESTEPS, DEVICE, PROJECT_DIR
from data_loader import build_dataset, prepare_data_splits
from model import count_parameters
from features_v6 import build_params_v6, N_V6_FEATURES
from anchor_features import build_bank, anchors_for
from train_v3 import (
    EvalDataset, EMA,
    LAMBDA_PEAK_WEIGHT, LAMBDA_PHYSICS_INIT, LAMBDA_SMOOTH, LAMBDA_REL,
    HIDDEN_SIZE, EMBEDDING_DIM, N_HEADS, DROPOUT, LEARNING_RATE, WEIGHT_DECAY,
    BATCH_SIZE, NUM_EPOCHS, PATIENCE, NUM_KNOTS, N_CANDIDATES, N_KEEP,
    pooled_r2,
)
from train_v5 import (
    PhysicsLossV5, train_epoch, validate,
    LAMBDA_GROWTH, LAMBDA_DECAY, LAMBDA_ENERGY, GROWTH_END_IDX, DECAY_START_IDX,
)
from train_v6 import AugDatasetV6, cladding_loo_resid, member_preds_degC
from train_v7 import make_model  # KANAttentionLSTMv7 (ambient clamp) + set_ambient

N_CONTINUOUS = 2 + 13 + N_V6_FEATURES  # 38
MODEL_DIR = os.path.join(PROJECT_DIR, "models_v8")

TAU_TAIL = 1.0   # scaled-residual scale for the adaptive tail-downweight


class PhysicsLossV8(PhysicsLossV5):
    """v5 loss with the data term robustified by an adaptive tail-downweight.

    Identical to PhysicsLossV5 except the peak-weighted MSE gains a detached
    per-point factor tw = 1/(1+(|e|/TAU)^2) that sheds gradient from the
    heavy-tail (chaotic-spike) points. All other terms unchanged.
    """

    def __init__(self, init_temp, scaler_mean, scaler_scale, tau=TAU_TAIL):
        super().__init__(init_temp, scaler_mean, scaler_scale)
        self.tau = float(tau)

    def forward(self, pred, target, masks=None, hrr=None):
        if masks is not None:
            me = masks.unsqueeze(-1)
        else:
            me = torch.ones_like(pred[:, :, :1])
        nv = me.sum().clamp(min=1.0)

        err2 = (pred - target) ** 2
        at = torch.abs(target) * me
        mt = at.max().clamp(min=1.0)
        w = 1.0 + LAMBDA_PEAK_WEIGHT * (at / mt)
        with torch.no_grad():
            e = (pred - target).abs()
            tw = 1.0 / (1.0 + (e / self.tau) ** 2)   # detached tail-downweight
        wmse = (err2 * w * tw * me).sum() / nv

        init_l = ((pred[:, 0, :] - self.init_temp.unsqueeze(0)) ** 2).mean()

        diff = pred[:, 1:, :] - pred[:, :-1, :]
        if masks is not None:
            dm = (masks[:, 1:] * masks[:, :-1]).unsqueeze(-1)
            sm = (diff ** 2 * dm).sum() / dm.sum().clamp(min=1.0)
        else:
            sm = (diff ** 2).mean()

        targ_degC = target * self.s_scale + self.s_mean
        hot = (targ_degC > 100.0).float() * me
        abs_degC = (pred - target).abs() * self.s_scale
        rel = (abs_degC / targ_degC.clamp(min=100.0) * hot).sum() / hot.sum().clamp(min=1.0)

        base = (wmse + LAMBDA_PHYSICS_INIT * init_l + LAMBDA_SMOOTH * sm
                + LAMBDA_REL * rel)

        # v5 causal physics terms (unchanged)
        dg = pred[:, 1:GROWTH_END_IDX, :] - pred[:, :GROWTH_END_IDX - 1, :]
        growth = F.relu(-dg).pow(2).mean()
        dd = pred[:, DECAY_START_IDX + 1:, :] - pred[:, DECAY_START_IDX:-1, :]
        decay = F.relu(dd).pow(2).mean()
        energy = pred.new_tensor(0.0)
        if hrr is not None and hrr.numel() > 2:
            mean_t = pred.mean(dim=(1, 2))
            h = hrr - hrr.mean()
            mm = mean_t - mean_t.mean()
            denom = h.std() * mm.std()
            if denom > 1e-6:
                energy = 1.0 - (h * mm).mean() / denom

        total = (base + LAMBDA_GROWTH * growth + LAMBDA_DECAY * decay
                 + LAMBDA_ENERGY * energy)
        return total, {}


def main(model_dir=MODEL_DIR, tau=TAU_TAIL):
    print("=" * 70)
    print("  BS8414 KAN-Attention-LSTM Surrogate - Training v8 (clamp+robust)")
    print("=" * 70)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.backends.cudnn.benchmark = True

    MODEL_DIR = model_dir
    os.makedirs(MODEL_DIR, exist_ok=True)

    params, outputs, masks, meta, sensor_names = build_dataset()
    _, _, _, scaler, split_info, time_array = \
        prepare_data_splits(params, outputs, masks, meta)

    train_idx = split_info["train_idx"]
    valid_idx = split_info["valid_idx"]
    test_idx = split_info["test_idx"]

    bank = build_bank(params, outputs, train_idx)
    np.savez(os.path.join(MODEL_DIR, "anchor_bank.npz"), **bank)

    params_v6 = build_params_v6(params, meta, bank)
    print(f"\n  Input vector: {params_v6.shape[1]} features "
          f"({len(params)} sims: {len(train_idx)}/{len(valid_idx)}/{len(test_idx)})")
    print(f"  Robust data loss: tail-downweight TAU={tau} (scaled); ambient clamp on")

    np.save(os.path.join(MODEL_DIR, "output_scaler_mean.npy"), scaler.mean_)
    np.save(os.path.join(MODEL_DIR, "output_scaler_scale.npy"), scaler.scale_)
    np.save(os.path.join(MODEL_DIR, "sensor_names.npy"), sensor_names)

    train_anchors = anchors_for(params[train_idx], bank,
                                loo_bank_pos=np.arange(len(train_idx)))
    valid_anchors = anchors_for(params[valid_idx], bank)
    test_anchors = anchors_for(params[test_idx], bank)

    loo = cladding_loo_resid(bank)
    jitter_std = np.array([max(0.02, loo[int(params[i, 0])])
                           for i in train_idx], dtype=np.float32)

    train_scaled = scaler.transform(
        outputs[train_idx].reshape(-1, N_SENSORS)
    ).reshape(-1, N_TIMESTEPS, N_SENSORS).astype(np.float32)
    valid_scaled = scaler.transform(
        outputs[valid_idx].reshape(-1, N_SENSORS)
    ).reshape(-1, N_TIMESTEPS, N_SENSORS).astype(np.float32)

    aug = AugDatasetV6(params_v6[train_idx], train_scaled, masks[train_idx],
                       time_array, train_anchors, jitter_std)
    train_loader = DataLoader(aug, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(
        EvalDataset(params_v6[valid_idx], valid_scaled, masks[valid_idx],
                    time_array, valid_anchors),
        batch_size=BATCH_SIZE)

    valid_actual = outputs[valid_idx]
    test_actual = outputs[test_idx]

    print(f"  Train: {len(train_idx)} real -> {len(aug)} augmented")

    init_temp = torch.FloatTensor(
        scaler.transform(np.full((1, N_SENSORS), 18.0, dtype=np.float32))[0]
    ).to(device)
    criterion = PhysicsLossV8(
        init_temp,
        torch.FloatTensor(scaler.mean_), torch.FloatTensor(scaler.scale_),
        tau=tau,
    ).to(device)

    print(f"\n--- Training {N_CANDIDATES} candidate members ---")
    members, val_losses, valid_pred_stack = [], [], []
    t0 = time_mod.time()

    for m_idx in range(N_CANDIDATES):
        seed = 42 + m_idx * 100
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        model = make_model(device, scaler.mean_, scaler.scale_)
        if m_idx == 0:
            print(f"  Params: {count_parameters(model):,}")

        opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=75, T_mult=2, eta_min=1e-5)
        ema = EMA(model, decay=0.999)
        eval_model = make_model(device, scaler.mean_, scaler.scale_)

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
        vp = member_preds_degC(model, params_v6[valid_idx], valid_anchors,
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

    print(f"\n--- Greedy ensemble selection (max {N_KEEP} members, by valid R2) ---")
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

    for name, sp, sa, an in [("Valid", params_v6[valid_idx], valid_actual, valid_anchors),
                             ("Test", params_v6[test_idx], test_actual, test_anchors)]:
        stack = np.stack([member_preds_degC(m, sp, an, time_array, scaler, device)
                          for m in ensemble])
        ens = np.tensordot(weights, stack, axes=(0, 0))
        r2 = r2_score(sa.flatten(), ens.flatten())
        rmse = np.sqrt(mean_squared_error(sa.flatten(), ens.flatten()))
        hot = sa.flatten() > 100
        mape = np.mean(np.abs((sa.flatten()[hot] - ens.flatten()[hot])
                              / sa.flatten()[hot])) * 100
        sub = (ens.flatten() < 17.0).sum()
        print(f"\n  {name}: R2={r2:.4f}  RMSE={rmse:.1f}C  MAPE(T>100)={mape:.1f}%  "
              f"sub-ambient pts={sub}")
        for k in range(len(sa)):
            print(f"    sim {k}: R2={r2_score(sa[k].flatten(), ens[k].flatten()):.3f}")

    print(f"\n  Saved -> {os.path.join(MODEL_DIR, 'best_model.pt')}")


if __name__ == "__main__":
    main()
