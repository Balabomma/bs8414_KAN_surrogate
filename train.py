"""Training for KAN-Attention-LSTM surrogate model."""
import os, shutil, time as time_mod, copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from config import N_SENSORS, N_TIMESTEPS, DEVICE, MODEL_DIR, OUTPUT_DIR
from data_loader import build_dataset, prepare_data_splits, FDSSurrogateDataset
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
N_ENSEMBLE = 5
NUM_KNOTS = 8

LAMBDA_PHYSICS_INIT = 0.1
LAMBDA_SMOOTH = 0.015
LAMBDA_PEAK_WEIGHT = 0.5
LAMBDA_KAN_REG = 1e-3  # L2 + smoothness on spline weights
EMA_DECAY = 0.999


class AugDataset(Dataset):
    def __init__(self, params, outputs, masks, time_array, augment=True):
        self.params = torch.FloatTensor(params)
        self.outputs = torch.FloatTensor(outputs)
        self.masks = torch.FloatTensor(masks)
        self.time_array = torch.FloatTensor(time_array)
        self.augment = augment
    def __len__(self):
        return len(self.params) * 4 if self.augment else len(self.params)
    def __getitem__(self, idx):
        n = len(self.params)
        if not self.augment or idx < n:
            i = idx % n
            return self.params[i], self.outputs[i], self.masks[i], self.time_array
        elif idx < n*2:
            i = idx-n
            return self.params[i], self.outputs[i]+torch.randn_like(self.outputs[i])*0.03, self.masks[i], self.time_array
        elif idx < n*3:
            i, j = idx-n*2, torch.randint(0,n,(1,)).item()
            lam = max(np.random.beta(0.3,0.3), 0.5)
            return lam*self.params[i]+(1-lam)*self.params[j], lam*self.outputs[i]+(1-lam)*self.outputs[j], torch.ones_like(self.masks[i]), self.time_array
        else:
            i = idx-n*3
            s = 0.85+torch.rand(1).item()*0.30
            b = self.outputs[i][0:1,:]
            return self.params[i], b+(self.outputs[i]-b)*s, self.masks[i], self.time_array


class PhysicsLoss(nn.Module):
    def __init__(self, init_temp):
        super().__init__()
        self.init_temp = init_temp
    def forward(self, pred, target, masks=None):
        if masks is not None:
            me = masks.unsqueeze(-1)
        else:
            me = torch.ones_like(pred[:,:,:1])
        nv = me.sum().clamp(min=1.0)
        errors = (pred-target)**2 * me
        at = torch.abs(target)*me
        mt = at.max().clamp(min=1.0)
        w = 1.0 + LAMBDA_PEAK_WEIGHT*(at/mt)
        wmse = (errors*w).sum()/nv
        init_l = ((pred[:,0,:]-self.init_temp.unsqueeze(0))**2).mean()
        diff = pred[:,1:,:]-pred[:,:-1,:]
        if masks is not None:
            dm = (masks[:,1:]*masks[:,:-1]).unsqueeze(-1)
            sm = (diff**2*dm).sum()/dm.sum().clamp(min=1.0)
        else:
            sm = (diff**2).mean()
        return wmse + LAMBDA_PHYSICS_INIT*init_l + LAMBDA_SMOOTH*sm, {}


class EMA:
    """Exponential moving average of model parameters for evaluation/checkpointing."""
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)
    def apply_to(self, model):
        """Return a state_dict that blends EMA (float tensors) with current buffers."""
        sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for k in self.shadow:
            sd[k] = self.shadow[k].clone()
        return sd


def train_epoch(model, loader, opt, criterion, device, ema=None):
    model.train(); total, n = 0, 0
    for p,t,m,ta in loader:
        p,t,m = p.to(device),t.to(device),m.to(device); ta=ta[0].to(device)
        opt.zero_grad()
        pred,_ = model(p,ta)
        loss,_ = criterion(pred,t,m)
        loss = loss + LAMBDA_KAN_REG * kan_regularization(model)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        opt.step()
        if ema is not None:
            ema.update(model)
        total+=loss.item(); n+=1
    return total/n


def validate(model, loader, criterion, device):
    model.eval(); total, n = 0, 0
    with torch.no_grad():
        for p,t,m,ta in loader:
            p,t,m = p.to(device),t.to(device),m.to(device); ta=ta[0].to(device)
            total+=criterion(model(p,ta)[0],t,m)[0].item(); n+=1
    return total/n


def main():
    print("="*70)
    print("  BS8414 KAN-Attention-LSTM Surrogate — Training")
    print("="*70)

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.backends.cudnn.benchmark = True

    for d in [MODEL_DIR, OUTPUT_DIR]:
        if os.path.exists(d): shutil.rmtree(d)
        os.makedirs(d)

    params, outputs, masks, meta, sensor_names = build_dataset()
    train_ds, valid_ds, test_ds, scaler, split_info, time_array = \
        prepare_data_splits(params, outputs, masks, meta)

    np.save(os.path.join(MODEL_DIR,"output_scaler_mean.npy"), scaler.mean_)
    np.save(os.path.join(MODEL_DIR,"output_scaler_scale.npy"), scaler.scale_)
    np.save(os.path.join(MODEL_DIR,"sensor_names.npy"), sensor_names)

    train_params = params[split_info["train_idx"]]
    train_scaled = scaler.transform(outputs[split_info["train_idx"]].reshape(-1,N_SENSORS)).reshape(-1,N_TIMESTEPS,N_SENSORS).astype(np.float32)
    train_masks = masks[split_info["train_idx"]]

    aug = AugDataset(train_params, train_scaled, train_masks, time_array)
    train_loader = DataLoader(aug, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE)

    print(f"\n  Train: {len(train_params)} real -> {len(aug)} augmented")

    init_temp = torch.FloatTensor(scaler.transform(np.full((1,N_SENSORS),18.0,dtype=np.float32))[0]).to(device)
    criterion = PhysicsLoss(init_temp)

    print(f"\n--- Training {N_ENSEMBLE} Ensemble ---")
    ensemble, val_losses = [], []
    t0 = time_mod.time()

    for m_idx in range(N_ENSEMBLE):
        seed = 42 + m_idx*100
        torch.manual_seed(seed); torch.cuda.manual_seed(seed); np.random.seed(seed)

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

        # Temporary model used for EMA-based validation (same architecture)
        eval_model = KANAttentionLSTM(
            n_sensors=N_SENSORS, hidden_size=HIDDEN_SIZE,
            embedding_dim=EMBEDDING_DIM, n_heads=N_HEADS,
            dropout=DROPOUT, num_knots=NUM_KNOTS,
        ).to(device)

        best_v, best_s, pat = float("inf"), None, 0
        for ep in range(1, NUM_EPOCHS+1):
            tl = train_epoch(model, train_loader, opt, criterion, device, ema=ema)
            # Validate with EMA weights for smoother, less-overfit early-stopping signal
            eval_model.load_state_dict(ema.apply_to(model))
            vl = validate(eval_model, valid_loader, criterion, device)
            sched.step()
            if ep % 100 == 0:
                print(f"    [M{m_idx+1}] Ep{ep:4d} T:{tl:.4f} V(ema):{vl:.4f}")
            if vl < best_v:
                best_v=vl; best_s=copy.deepcopy(eval_model.state_dict()); pat=0
            else:
                pat+=1
                if pat>=PATIENCE: break

        model.load_state_dict(best_s)
        ensemble.append(model); val_losses.append(best_v)
        print(f"    [M{m_idx+1}] Best valid: {best_v:.4f}")

    # Weighted ensemble
    weights = np.array([1.0/v for v in val_losses])
    weights = weights / weights.sum()
    print(f"\n  Weights: {[f'{w:.3f}' for w in weights]}")
    print(f"  Time: {time_mod.time()-t0:.0f}s")

    torch.save({
        "model_states": [m.state_dict() for m in ensemble],
        "n_models": N_ENSEMBLE, "ensemble_weights": weights.tolist(),
    }, os.path.join(MODEL_DIR, "best_model.pt"))

    # Evaluate
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    preds, targets = [], []
    with torch.no_grad():
        for p,t,m,ta in test_loader:
            p=p.to(device); ta=ta[0].to(device)
            bp = [model(p,ta)[0].cpu().numpy()*w for model,w in zip(ensemble,weights)]
            preds.append(sum(bp)); targets.append(t.numpy())

    pred = np.concatenate(preds); actual = np.concatenate(targets)
    pt = (pred.reshape(-1,N_SENSORS)*scaler.scale_+scaler.mean_).reshape(pred.shape)
    at = (actual.reshape(-1,N_SENSORS)*scaler.scale_+scaler.mean_).reshape(actual.shape)

    r2 = r2_score(at.flatten(), pt.flatten())
    rmse = np.sqrt(mean_squared_error(at.flatten(), pt.flatten()))
    mask100 = at.flatten() > 100
    mape = np.mean(np.abs((at.flatten()[mask100]-pt.flatten()[mask100])/at.flatten()[mask100]))*100

    sr2 = [r2_score(at[:,:,s].flatten(), pt[:,:,s].flatten()) for s in range(N_SENSORS)]

    print(f"\n{'='*70}")
    print(f"  KAN-Attention-LSTM RESULTS")
    print(f"{'='*70}")
    print(f"  Test R2:        {r2:.4f}")
    print(f"  Test RMSE:      {rmse:.1f} C")
    print(f"  MAPE (T>100C):  {mape:.1f}%")
    print(f"  Avg sensor R2:  {np.mean(sr2):.4f}")
    print(f"  Sensors R2>0.9: {sum(1 for s in sr2 if s>0.9)}/24")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
