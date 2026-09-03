"""Training v2 for KAN-Attention-LSTM surrogate — targeted at raising valid R².

Changes vs train.py (all model selection uses the VALID set only):
  1. Mixup augmentation restricted to same-cladding pairs. The old cross-cladding
     mixup averaged the integer cladding id (e.g. 0.6*2 + 0.4*1 = 1.6), which the
     encoder's .long() truncated to the wrong embedding, and blending spikes at
     different ignition times taught the model smeared twin-bump transients.
  2. LAMBDA_PEAK_WEIGHT 0.5 -> 1.0 and LAMBDA_SMOOTH 0.015 -> 0.005 so sharp
     ignition transients (Test_1_PE_PIR low-HRR family) are not penalised away.
  3. Best-of-N member selection: train N_CANDIDATES members, greedy-select the
     subset (max N_KEEP) that maximises pooled valid R², weights = 1/valid_loss.
  4. Saves to models_v2/ (production models/ untouched); checkpoint format is
     identical to train.py so evaluate_kan.py works via --model-dir models_v2.

Usage: python train_v2.py
"""
import os, time as time_mod, copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import r2_score, mean_squared_error

from config import N_SENSORS, N_TIMESTEPS, DEVICE, PROJECT_DIR
from data_loader import build_dataset, prepare_data_splits
from model import KANAttentionLSTM, count_parameters, kan_regularization

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

N_CANDIDATES = 12   # members trained
N_KEEP = 5          # max members kept (greedy by pooled valid R²)

LAMBDA_PHYSICS_INIT = 0.1
LAMBDA_SMOOTH = 0.005      # was 0.015
LAMBDA_PEAK_WEIGHT = 1.0   # was 0.5
LAMBDA_KAN_REG = 1e-3
EMA_DECAY = 0.999

MODEL_DIR = os.path.join(PROJECT_DIR, "models_v2")


class AugDataset(Dataset):
    """Same 4x augmentation as train.py, but mixup pairs stay within one cladding."""

    def __init__(self, params, outputs, masks, time_array):
        self.params = torch.FloatTensor(params)
        self.outputs = torch.FloatTensor(outputs)
        self.masks = torch.FloatTensor(masks)
        self.time_array = torch.FloatTensor(time_array)
        # index pool per cladding id for same-cladding mixup
        clad = params[:, 0].astype(int)
        self.same_clad = {i: np.where(clad == clad[i])[0] for i in range(len(params))}

    def __len__(self):
        return len(self.params) * 4

    def __getitem__(self, idx):
        n = len(self.params)
        if idx < n:
            i = idx
            return self.params[i], self.outputs[i], self.masks[i], self.time_array
        elif idx < n * 2:
            i = idx - n
            return self.params[i], self.outputs[i] + torch.randn_like(self.outputs[i]) * 0.03, self.masks[i], self.time_array
        elif idx < n * 3:
            i = idx - n * 2
            pool = self.same_clad[i]
            j = int(pool[torch.randint(0, len(pool), (1,)).item()])
            lam = max(np.random.beta(0.3, 0.3), 0.5)
            return (lam * self.params[i] + (1 - lam) * self.params[j],
                    lam * self.outputs[i] + (1 - lam) * self.outputs[j],
                    torch.ones_like(self.masks[i]), self.time_array)
        else:
            i = idx - n * 3
            s = 0.85 + torch.rand(1).item() * 0.30
            b = self.outputs[i][0:1, :]
            return self.params[i], b + (self.outputs[i] - b) * s, self.masks[i], self.time_array


class PhysicsLoss(nn.Module):
    def __init__(self, init_temp):
        super().__init__()
        self.init_temp = init_temp

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
        return wmse + LAMBDA_PHYSICS_INIT * init_l + LAMBDA_SMOOTH * sm, {}


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


def train_epoch(model, loader, opt, criterion, device, ema):
    model.train()
    total, n = 0, 0
    for p, t, m, ta in loader:
        p, t, m = p.to(device), t.to(device), m.to(device)
        ta = ta[0].to(device)
        opt.zero_grad()
        pred, _ = model(p, ta)
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
        for p, t, m, ta in loader:
            p, t, m = p.to(device), t.to(device), m.to(device)
            ta = ta[0].to(device)
            total += criterion(model(p, ta)[0], t, m)[0].item()
            n += 1
    return total / n


def member_preds_degC(model, params_np, time_array, scaler, device):
    """Predict all sims for one member, denormalised to degC."""
    model.eval()
    with torch.no_grad():
        p = torch.FloatTensor(params_np).to(device)
        ta = torch.FloatTensor(time_array).to(device)
        out = model(p, ta)[0].cpu().numpy()
    return (out.reshape(-1, N_SENSORS) * scaler.scale_ + scaler.mean_).reshape(out.shape)


def pooled_r2(pred_stack, weights, actual):
    """R² of weighted-mean ensemble prediction vs actual (both degC)."""
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum()
    ens = np.tensordot(w, pred_stack, axes=(0, 0))
    return r2_score(actual.flatten(), ens.flatten())


def main():
    print("=" * 70)
    print("  BS8414 KAN-Attention-LSTM Surrogate — Training v2 (best-of-N)")
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

    train_params = params[split_info["train_idx"]]
    train_scaled = scaler.transform(
        outputs[split_info["train_idx"]].reshape(-1, N_SENSORS)
    ).reshape(-1, N_TIMESTEPS, N_SENSORS).astype(np.float32)
    train_masks = masks[split_info["train_idx"]]

    aug = AugDataset(train_params, train_scaled, train_masks, time_array)
    train_loader = DataLoader(aug, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE)

    valid_params = params[split_info["valid_idx"]]
    valid_actual = outputs[split_info["valid_idx"]]
    test_params = params[split_info["test_idx"]]
    test_actual = outputs[split_info["test_idx"]]

    print(f"\n  Train: {len(train_params)} real -> {len(aug)} augmented")
    print(f"  Peak weight={LAMBDA_PEAK_WEIGHT}  smooth={LAMBDA_SMOOTH}  mixup=same-cladding")

    init_temp = torch.FloatTensor(
        scaler.transform(np.full((1, N_SENSORS), 18.0, dtype=np.float32))[0]
    ).to(device)
    criterion = PhysicsLoss(init_temp)

    print(f"\n--- Training {N_CANDIDATES} candidate members ---")
    members, val_losses, valid_pred_stack = [], [], []
    t0 = time_mod.time()

    for m_idx in range(N_CANDIDATES):
        seed = 42 + m_idx * 100
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

        model = KANAttentionLSTM(
            n_sensors=N_SENSORS, hidden_size=HIDDEN_SIZE,
            embedding_dim=EMBEDDING_DIM, n_heads=N_HEADS,
            dropout=DROPOUT, num_knots=NUM_KNOTS,
        ).to(device)
        if m_idx == 0:
            print(f"  Params: {count_parameters(model):,}")

        opt = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=75, T_mult=2, eta_min=1e-5)
        ema = EMA(model, decay=EMA_DECAY)
        eval_model = KANAttentionLSTM(
            n_sensors=N_SENSORS, hidden_size=HIDDEN_SIZE,
            embedding_dim=EMBEDDING_DIM, n_heads=N_HEADS,
            dropout=DROPOUT, num_knots=NUM_KNOTS,
        ).to(device)

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
        vp = member_preds_degC(model, valid_params, time_array, scaler, device)
        valid_pred_stack.append(vp)
        vr2 = r2_score(valid_actual.flatten(), vp.flatten())
        print(f"  [M{m_idx+1:2d}] valid loss={best_v:.4f}  valid R2={vr2:.4f}  "
              f"({time_mod.time()-t0:.0f}s elapsed)")

    valid_pred_stack = np.stack(valid_pred_stack)  # (N, n_valid, T, S)

    # ── Greedy forward selection on pooled valid R² (weights = 1/valid_loss) ──
    print(f"\n--- Greedy ensemble selection (max {N_KEEP} members, by valid R²) ---")
    selected, best_r2 = [], -np.inf
    while len(selected) < N_KEEP:
        best_cand, best_cand_r2 = None, best_r2
        for c in range(N_CANDIDATES):
            if c in selected:
                continue
            trial = selected + [c]
            r2 = pooled_r2(valid_pred_stack[trial], [1.0 / val_losses[t] for t in trial], valid_actual)
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

    # All candidates too, so ensembles can be re-selected/re-weighted without retraining
    torch.save({
        "model_states": [m.state_dict() for m in members],
        "n_models": len(members),
        "val_losses": val_losses,
    }, os.path.join(MODEL_DIR, "all_candidates.pt"))

    # ── Final metrics (valid = selection set, test = held out, reported once) ──
    for name, sp, sa in [("Valid", valid_params, valid_actual), ("Test", test_params, test_actual)]:
        stack = np.stack([member_preds_degC(m, sp, time_array, scaler, device) for m in ensemble])
        ens = np.tensordot(weights, stack, axes=(0, 0))
        r2 = r2_score(sa.flatten(), ens.flatten())
        rmse = np.sqrt(mean_squared_error(sa.flatten(), ens.flatten()))
        print(f"\n  {name}: R2={r2:.4f}  RMSE={rmse:.1f}C")
        for k in range(len(sa)):
            sr2 = r2_score(sa[k].flatten(), ens[k].flatten())
            print(f"    sim {k}: R2={sr2:.3f}")

    print(f"\n  Saved -> {os.path.join(MODEL_DIR, 'best_model.pt')}")


if __name__ == "__main__":
    main()
