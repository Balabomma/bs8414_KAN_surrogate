"""Ensemble every trained member across runs, in physical space."""
import glob, os, sys, json
import numpy as np, torch
from config_part1 import HRR_CHANNELS
from data_loader_part1 import build_dataset, prepare_data_splits, ChannelScaler
from evaluate_part1 import masked_r2, masked_rmse
from model_part1 import Part1Surrogate

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
params, tc, hrr, mask, meta, _ = build_dataset(verbose=False)
ds, _, _, info, ta = prepare_data_splits(params, tc, hrr, mask, meta, mode="hash")

EXCLUDE = ("phys", "sys_")            # different objective / different split
ckpts = []
for d in sorted(glob.glob("models_part1*")):
    if any(x in d for x in EXCLUDE): continue
    for f in sorted(glob.glob(os.path.join(d, "part1_member*.pt"))):
        ckpts.append(f)
print(f"members found: {len(ckpts)} across {len(set(os.path.dirname(c) for c in ckpts))} runs")

@torch.no_grad()
def preds_for(split):
    d = ds[split]; P = d.params.to(dev); T = d.time_array.to(dev)
    acc, n_ok = None, 0
    per_run = {}
    for f in ckpts:
        ck = torch.load(f, map_location=dev, weights_only=False)
        m = Part1Surrogate().to(dev); m.load_state_dict(ck["state_dict"]); m.eval()
        s = ChannelScaler().load_state_dict(ck["tc_scaler"])
        hs = ChannelScaler().load_state_dict(ck["hrr_scaler"])
        m.set_output_scaling(s, hs, hrr_nonnegative_idx=[HRR_CHANNELS.index("HRR")])
        out = s.inverse(m(P, T)[0].cpu().numpy())
        acc = out if acc is None else acc + out
        n_ok += 1
        per_run.setdefault(os.path.dirname(f), []).append(out)
    return acc / n_ok, per_run, d

for split in ("valid", "test"):
    mean_pred, per_run, d = preds_for(split)
    true = ChannelScaler().load_state_dict(
        torch.load(ckpts[0], map_location=dev, weights_only=False)["tc_scaler"]
    ).inverse(d.tc.numpy())
    mk = d.mask.numpy()
    print(f"  [{split}] ALL-MEMBER ensemble ({len(ckpts)} members): "
          f"R2 {masked_r2(mean_pred, true, mk):.4f}  RMSE {masked_rmse(mean_pred, true, mk):.2f}")
    # per-run ensembles for reference
    best = sorted(((masked_r2(np.mean(v,0), true, mk), k) for k,v in per_run.items()), reverse=True)
    print(f"        best single run: {best[0][0]:.4f} ({os.path.basename(best[0][1])})")
    print(f"        worst single run: {best[-1][0]:.4f} ({os.path.basename(best[-1][1])})")
