"""Training v3 for KAN-Attention-LSTM surrogate — anchor-time channels.

Builds on train_v2 (same-cladding mixup, peak weight 1.0, best-of-N greedy
ensemble selection on valid R²) and adds:

  1. Ignition-anchor input channels (model_v3.KANAttentionLSTMv3): the expected
     ignition-peak time is predicted from (cladding, hrr, mesh) by a
     mesh-localised 1/hrr regression fit on TRAIN sims only
     (anchor_features.py). Train sims use leave-one-out anchors + jitter so the
     network treats the hint as soft.
  2. Mixup pairs additionally restricted to anchor-compatible sims
     (|Δanchor| < 0.08) so mixed curves stay consistent with the mixed hint.
  3. Relative-error loss term on the hot region (targets MAPE(T>100) < 15%).
  4. Saves best_model.pt AND all_candidates.pt + anchor_bank.npz to models_v3/.

Usage: python train_v3.py
"""
import os, time as time_mod, copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score, mean_squared_error

from config import N_SENSORS, N_TIMESTEPS, DEVICE, PROJECT_DIR
from data_loader import build_dataset, prepare_data_splits
from model import count_parameters, kan_regularization
from model_v3 import KANAttentionLSTMv3
from anchor_features import build_bank, anchors_for

HIDDEN_SIZE = 96
EMBEDDING_DIM = 24
N_HEADS = 4
DROPOUT = 0.15
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 2e-4
BATCH_SIZE = 8
NUM_EPOCHS = 1500
PATIENCE = 200
NUM_KNOTS = 8

N_CANDIDATES = 12
N_KEEP = 5

LAMBDA_PHYSICS_INIT = 0.1
LAMBDA_SMOOTH = 0.005
LAMBDA_PEAK_WEIGHT = 1.0
LAMBDA_KAN_REG = 1e-3
LAMBDA_REL = 0.5           # relative-error (MAPE) term, hot region only
ANCHOR_JITTER = 0.02       # std of train-time anchor jitter (0.02*1800 = 36 s)
MIXUP_ANCHOR_TOL = 0.08    # only mix sims whose anchors differ < 144 s
EMA_DECAY = 0.999

MODEL_DIR = os.path.join(PROJECT_DIR, "models_v3")


class AugDatasetV3(Dataset):
    """4x augmentation; mixup restricted to same cladding AND compatible anchor."""

    def __init__(self, params, outputs, masks, time_array, anchors):
        self.params = torch.FloatTensor(params)
        self.outputs = torch.FloatTensor(outputs)
        self.masks = torch.FloatTensor(masks)
        self.time_array = torch.FloatTensor(time_array)
        self.anchors = torch.FloatTensor(anchors)
        clad = params[:, 0].astype(int)
        self.pool = {}
        for i in range(len(params)):
            ok = np.where((clad == clad[i]) &
                          (np.abs(anchors - anchors[i]) < MIXUP_ANCHOR_TOL))[0]
            self.pool[i] = ok if len(ok) > 0 else np.array([i])

    def __len__(self):
        return len(self.params) * 4

    def _jit(self, a):
        return a + torch.randn(1).item() * ANCHOR_JITTER

    def __getitem__(self, idx):
        n = len(self.params)
        if idx < n:
            i = idx
            return (self.params[i], self.outputs[i], self.masks[i],
                    self.time_array, self._jit(self.anchors[i]))
        elif idx < n * 2:
            i = idx - n
            return (self.params[i],
                    self.outputs[i] + torch.randn_like(self.outputs[i]) * 0.03,
                    self.masks[i], self.time_array, self._jit(self.anchors[i]))
        elif idx < n * 3:
            i = idx - n * 2
            pool = self.pool[i]
            j = int(pool[torch.randint(0, len(pool), (1,)).item()])
            lam = max(np.random.beta(0.3, 0.3), 0.5)
            return (lam * self.params[i] + (1 - lam) * self.params[j],
                    lam * self.outputs[i] + (1 - lam) * self.outputs[j],
                    torch.ones_like(self.masks[i]), self.time_array,
                    self._jit(lam * self.anchors[i] + (1 - lam) * self.anchors[j]))
        else:
            i = idx - n * 3
            s = 0.85 + torch.rand(1).item() * 0.30
            b = self.outputs[i][0:1, :]
            return (self.params[i], b + (self.outputs[i] - b) * s,
                    self.masks[i], self.time_array, self._jit(self.anchors[i]))


class EvalDataset(Dataset):
    def __init__(self, params, outputs, masks, time_array, anchors):
        self.params = torch.FloatTensor(params)
        self.outputs = torch.FloatTensor(outputs)
        self.masks = torch.FloatTensor(masks)
        self.time_array = torch.FloatTensor(time_array)
        self.anchors = torch.FloatTensor(anchors)

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        return (self.params[idx], self.outputs[idx], self.masks[idx],
                self.time_array, self.anchors[idx])


class PhysicsLossV3(nn.Module):
    """v2 loss + relative-error term on hot region (drives MAPE down)."""

    def __init__(self, init_temp, scaler_mean, scaler_scale):
        super().__init__()
        self.init_temp = init_temp
        self.register_buffer("s_mean", scaler_mean)    # (S,)
        self.register_buffer("s_scale", scaler_scale)  # (S,)

    def forward(self, pred, target, masks=None):
        if masks is not None:
            me = masks.unsqueeze(-1)
        else:
            me = torch.ones_like(pred[:, :, :1])
        nv = me.sum().clamp(min=1.0)
        errors = (pred - target) ** 2 * me
        at = torch.abs(target) * me
        mt = at.max().clamp(min=1.0)
        w = 1.0 + LAMBDA_PEAK_WEIGHT * (at / mt)
        wmse = (errors * w).sum() / nv

        init_l = ((pred[:, 0, :] - self.init_temp.unsqueeze(0)) ** 2).mean()

        diff = pred[:, 1:, :] - pred[:, :-1, :]
        if masks is not None:
            dm = (masks[:, 1:] * masks[:, :-1]).unsqueeze(-1)
            sm = (diff ** 2 * dm).sum() / dm.sum().clamp(min=1.0)
        else:
            sm = (diff ** 2).mean()

        # relative error where actual temperature > 100 degC
        targ_degC = target * self.s_scale + self.s_mean
        hot = (targ_degC > 100.0).float() * me
        abs_degC = (pred - target).abs() * self.s_scale
        rel = (abs_degC / targ_degC.clamp(min=100.0) * hot).sum() / hot.sum().clamp(min=1.0)

        total = (wmse + LAMBDA_PHYSICS_INIT * init_l + LAMBDA_SMOOTH * sm
                 + LAMBDA_REL * rel)
        return total, {}


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def apply_to(self, model):
        sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for k in self.shadow:
            sd[k] = self.shadow[k].clone()
        return sd


def make_model(device):
    return KANAttentionLSTMv3(
        n_sensors=N_SENSORS, hidden_size=HIDDEN_SIZE,
        embedding_dim=EMBEDDING_DIM, n_heads=N_HEADS,
        dropout=DROPOUT, num_knots=NUM_KNOTS,
    ).to(device)


def train_epoch(model, loader, opt, criterion, device, ema):
    model.train()
    total, n = 0, 0
    for p, t, m, ta, a in loader:
        p, t, m, a = p.to(device), t.to(device), m.to(device), a.to(device)
        ta = ta[0].to(device)
        opt.zero_grad()
        pred, _ = model(p, ta, a)
        loss, _ = criterion(pred, t, m)
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
            total += criterion(model(p, ta, a)[0], t, m)[0].item()
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


def pooled_r2(pred_stack, weights, actual):
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    ens = np.tensordot(w, pred_stack, axes=(0, 0))
    return r2_score(actual.flatten(), ens.flatten())


def main():
    print("=" * 70)
    print("  BS8414 KAN-Attention-LSTM Surrogate — Training v3 (anchor channels)")
    print("=" * 70)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.backends.cudnn.benchmark = True

    os.makedirs(MODEL_DIR, exist_ok=True)

    params, outputs, masks, meta, sensor_names = build_dataset()
    train_ds, valid_ds, test_ds, scaler, split_info, time_array = \
        prepare_data_splits(params, outputs, masks, meta)

    np.save(os.path.join(MODEL_DIR, "output_scaler_mean.npy"), scaler.mean_)
    np.save(os.path.join(MODEL_DIR, "output_scaler_scale.npy"), scaler.scale_)
    np.save(os.path.join(MODEL_DIR, "sensor_names.npy"), sensor_names)

    train_idx = split_info["train_idx"]
    valid_idx = split_info["valid_idx"]
    test_idx = split_info["test_idx"]

    # ── Anchor bank (train sims only) + predictions ──
    bank = build_bank(params, outputs, train_idx)
    np.savez(os.path.join(MODEL_DIR, "anchor_bank.npz"), **bank)

    loo_pos = np.arange(len(train_idx))
    train_anchors = anchors_for(params[train_idx], bank, loo_bank_pos=loo_pos)
    valid_anchors = anchors_for(params[valid_idx], bank)
    test_anchors = anchors_for(params[test_idx], bank)
    actual_train_anchors = bank["anchor"]
    print(f"\n  Anchor LOO error (train): mean|err|="
          f"{np.abs(train_anchors - actual_train_anchors).mean() * 1800:.0f}s")

    train_params = params[train_idx]
    train_scaled = scaler.transform(
        outputs[train_idx].reshape(-1, N_SENSORS)
    ).reshape(-1, N_TIMESTEPS, N_SENSORS).astype(np.float32)
    train_masks = masks[train_idx]

    valid_scaled = scaler.transform(
        outputs[valid_idx].reshape(-1, N_SENSORS)
    ).reshape(-1, N_TIMESTEPS, N_SENSORS).astype(np.float32)

    aug = AugDatasetV3(train_params, train_scaled, train_masks, time_array, train_anchors)
    train_loader = DataLoader(aug, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(
        EvalDataset(params[valid_idx], valid_scaled, masks[valid_idx], time_array, valid_anchors),
        batch_size=BATCH_SIZE)

    valid_actual = outputs[valid_idx]
    test_actual = outputs[test_idx]

    print(f"  Train: {len(train_params)} real -> {len(aug)} augmented")
    print(f"  rel-loss={LAMBDA_REL}  anchor jitter={ANCHOR_JITTER * 1800:.0f}s  "
          f"mixup tol={MIXUP_ANCHOR_TOL * 1800:.0f}s")

    init_temp = torch.FloatTensor(
        scaler.transform(np.full((1, N_SENSORS), 18.0, dtype=np.float32))[0]
    ).to(device)
    criterion = PhysicsLossV3(
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

        opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=75, T_mult=2, eta_min=1e-5)
        ema = EMA(model, decay=EMA_DECAY)
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
        vp = member_preds_degC(model, params[valid_idx], valid_anchors, time_array, scaler, device)
        valid_pred_stack.append(vp)
        vr2 = r2_score(valid_actual.flatten(), vp.flatten())
        print(f"  [M{m_idx+1:2d}] valid loss={best_v:.4f}  valid R2={vr2:.4f}  "
              f"({time_mod.time()-t0:.0f}s elapsed)")

    valid_pred_stack = np.stack(valid_pred_stack)

    # save every candidate for post-hoc re-selection
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
    print(f"  Selected: {[f'M{i+1}' for i in selected]}  weights={[f'{w:.3f}' for w in weights]}")
    print(f"  Time: {time_mod.time()-t0:.0f}s")

    torch.save({
        "model_states": [m.state_dict() for m in ensemble],
        "n_models": len(ensemble),
        "ensemble_weights": weights.tolist(),
        "selected_candidates": [i + 1 for i in selected],
        "candidate_val_losses": val_losses,
    }, os.path.join(MODEL_DIR, "best_model.pt"))

    for name, sp, sa, an in [("Valid", params[valid_idx], valid_actual, valid_anchors),
                             ("Test", params[test_idx], test_actual, test_anchors)]:
        stack = np.stack([member_preds_degC(m, sp, an, time_array, scaler, device)
                          for m in ensemble])
        ens = np.tensordot(weights, stack, axes=(0, 0))
        r2 = r2_score(sa.flatten(), ens.flatten())
        rmse = np.sqrt(mean_squared_error(sa.flatten(), ens.flatten()))
        hot = sa.flatten() > 100
        mape = np.mean(np.abs((sa.flatten()[hot] - ens.flatten()[hot]) / sa.flatten()[hot])) * 100
        print(f"\n  {name}: R2={r2:.4f}  RMSE={rmse:.1f}C  MAPE(T>100)={mape:.1f}%")
        for k in range(len(sa)):
            sr2 = r2_score(sa[k].flatten(), ens[k].flatten())
            print(f"    sim {k}: R2={sr2:.3f}")

    print(f"\n  Saved -> {os.path.join(MODEL_DIR, 'best_model.pt')}")


if __name__ == "__main__":
    main()
