"""BR 135 external fire-spread classification on Part1: surrogate vs FDS.

t_s  = first time mean External LV1 exceeds ambient + 200 degC (BS 8414 start)
FAIL = any External LV2 TC >= ambient + 600 degC sustained >= 30 s (3 samples)
       within 900 s of t_s
Part1 instruments the external face only, so this external criterion is fully
computable; the BR 135 *internal* criterion is not (no Insulation LV2 group).
"""
import glob, os
import numpy as np, torch
from config_part1 import HRR_CHANNELS, T_AMBIENT, DT_DEVC
from data_loader_part1 import build_dataset, prepare_data_splits, ChannelScaler
from model_part1 import Part1Surrogate

START_RISE, FAIL_RISE, SUSTAIN, WINDOW = 200.0, 600.0, 3, int(900/DT_DEVC)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
params, tc, hrr, mask, meta, _ = build_dataset(verbose=False)
ds,_,_,info,ta = prepare_data_splits(params, tc, hrr, mask, meta, mode="hash")
d = ds["test"]

# greedy-selected KAN ensemble from the previous step
SEL = ["models_part1_r4_seed48/part1_member1.pt",
       "models_part1_r4_seed48/part1_member1.pt",
       "models_part1_r3_seed45/part1_member0.pt"]
P = d.params.to(dev); T = d.time_array.to(dev); acc=None
for f in SEL:
    ck=torch.load(f,map_location=dev,weights_only=False)
    m=Part1Surrogate().to(dev); m.load_state_dict(ck["state_dict"]); m.eval()
    s=ChannelScaler().load_state_dict(ck["tc_scaler"]); hs=ChannelScaler().load_state_dict(ck["hrr_scaler"])
    m.set_output_scaling(s,hs,hrr_nonnegative_idx=[HRR_CHANNELS.index("HRR")])
    with torch.no_grad(): o=s.inverse(m(P,T)[0].cpu().numpy())
    acc = o if acc is None else acc+o
pred = acc/len(SEL)
true = ChannelScaler().load_state_dict(ck["tc_scaler"]).inverse(d.tc.numpy())
mk = d.mask.numpy()

def classify(arr, m):
    lv1 = arr[:, :8].mean(axis=1); lv2 = arr[:, 8:]
    ok = m > 0
    idx = np.where(ok & (lv1 >= T_AMBIENT + START_RISE))[0]
    if len(idx)==0: return False, None
    ts = idx[0]
    hi = (lv2 >= T_AMBIENT + FAIL_RISE)
    for j in range(lv2.shape[1]):
        run = 0
        for t in range(ts, min(ts+WINDOW+1, lv2.shape[0])):
            if not ok[t]: run = 0; continue
            run = run+1 if hi[t, j] else 0
            if run >= SUSTAIN: return True, ts
    return False, ts

agree = 0; rows=[]
for i in range(len(d.params)):
    ft,_ = classify(true[i], mk[i]); fp,_ = classify(pred[i], mk[i])
    agree += (ft==fp); rows.append((meta[i] if False else i, ft, fp))
n=len(rows)
tp=sum(1 for _,a,b in rows if a and b); tn=sum(1 for _,a,b in rows if not a and not b)
fp_=sum(1 for _,a,b in rows if not a and b); fn=sum(1 for _,a,b in rows if a and not b)
print(f"BR 135 EXTERNAL classification, {n} held-out test simulations")
print(f"  FDS says FAIL: {sum(1 for _,a,_ in rows if a)}   FDS says PASS: {sum(1 for _,a,_ in rows if not a)}")
print(f"  surrogate agreement: {agree}/{n}  ({100*agree/n:.1f}%)")
print(f"  confusion: TP {tp}  TN {tn}  FP {fp_}  FN {fn}")
print(f"  FN are the dangerous ones (FDS fails, surrogate says pass): {fn}")
