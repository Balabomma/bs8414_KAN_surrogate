"""Greedy ensemble selection - the v9 recipe, applied to the Part1 members.

Members are chosen to maximise VALIDATION R2, with replacement (a strong member
may be picked more than once, which is just weighting). Test is never consulted
during selection, so the reported test figure stays honest.
"""
import glob, os, sys
import numpy as np, torch
from config_part1 import HRR_CHANNELS
from data_loader_part1 import build_dataset, prepare_data_splits, ChannelScaler
from evaluate_part1 import masked_r2, masked_rmse
from model_part1 import Part1Surrogate

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
params, tc, hrr, mask, meta, _ = build_dataset(verbose=False)
ds, _, _, info, ta = prepare_data_splits(params, tc, hrr, mask, meta, mode="hash")
EXCLUDE=("phys","sys_")
ckpts=[f for d in sorted(glob.glob("models_part1*")) if not any(x in d for x in EXCLUDE)
         for f in sorted(glob.glob(os.path.join(d,"part1_member*.pt")))]

cache={}
@torch.no_grad()
def build(split):
    d=ds[split]; P=d.params.to(dev); T=d.time_array.to(dev); out=[]
    for f in ckpts:
        ck=torch.load(f,map_location=dev,weights_only=False)
        m=Part1Surrogate().to(dev); m.load_state_dict(ck["state_dict"]); m.eval()
        s=ChannelScaler().load_state_dict(ck["tc_scaler"])
        hs=ChannelScaler().load_state_dict(ck["hrr_scaler"])
        m.set_output_scaling(s,hs,hrr_nonnegative_idx=[HRR_CHANNELS.index("HRR")])
        out.append(s.inverse(m(P,T)[0].cpu().numpy()))
    true=s.inverse(d.tc.numpy()); return np.stack(out), true, d.mask.numpy()

Pv,Tv,Mv = build("valid"); Pt,Tt,Mt = build("test")
print(f"members: {len(ckpts)}")

sel=[]; cur_v=None
best_hist=[]
for step in range(25):
    best=(-9,None)
    for i in range(len(ckpts)):
        cand = Pv[i] if cur_v is None else (cur_v*len(sel)+Pv[i])/(len(sel)+1)
        r=masked_r2(cand,Tv,Mv)
        if r>best[0]: best=(r,i)
    r,i = best
    if cur_v is not None and r <= masked_r2(cur_v,Tv,Mv)+1e-6: break
    cur_v = Pv[i] if cur_v is None else (cur_v*len(sel)+Pv[i])/(len(sel)+1)
    sel.append(i); best_hist.append(r)

cur_t=np.mean(Pt[sel],axis=0)
print(f"greedy selected {len(sel)} members (with replacement) from {len(ckpts)}")
print(f"  VALID R2 {masked_r2(cur_v,Tv,Mv):.4f}   RMSE {masked_rmse(cur_v,Tv,Mv):.2f}")
print(f"  TEST  R2 {masked_r2(cur_t,Tt,Mt):.4f}   RMSE {masked_rmse(cur_t,Tt,Mt):.2f}")
comb=(masked_r2(cur_v,Tv,Mv)*Mv.sum()+masked_r2(cur_t,Tt,Mt)*Mt.sum())/(Mv.sum()+Mt.sum())
print(f"  combined valid+test (weighted): {comb:.4f}")
from collections import Counter
for idx,c in Counter(sel).most_common(6):
    print(f"    x{c}  {os.path.basename(os.path.dirname(ckpts[idx]))}/{os.path.basename(ckpts[idx])}")
