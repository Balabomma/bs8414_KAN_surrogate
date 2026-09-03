"""Does the GROUND TRUTH pass the growth-monotonicity gate?

`evaluate_part1.physics_gates` flags a prediction when any thermocouple falls more
than 5 degC between consecutive steps during the 0-720 s growth phase. If the FDS
data itself violates that, the gate is mis-calibrated and a FAIL says nothing about
the model. Measured before any claim is made either way.
"""
import numpy as np

from config_part1 import DT_DEVC, N_SENSORS
from data_loader_part1 import build_dataset, split_indices

GROWTH_END_STEP = int(720.0 / DT_DEVC)

params, tc, hrr, mask, meta, names = build_dataset(verbose=False)
idx = split_indices(meta)

for name in ("train", "valid", "test"):
    sel = idx[name]
    growth = tc[sel][:, :GROWTH_END_STEP + 1, :]
    d = np.diff(growth, axis=1)

    worst = float(d.min())
    n_below_5 = int((d < -5.0).sum())
    frac = n_below_5 / d.size
    pct = [float(np.percentile(d, q)) for q in (0.1, 1.0, 50.0)]

    print(f"\n[{name}]  {len(sel)} sims, ground-truth growth-phase step diffs")
    print(f"   worst single drop      {worst:>9.2f} degC")
    print(f"   steps below -5 degC    {n_below_5:>9} of {d.size} ({frac:.3%})")
    print(f"   percentiles  0.1% {pct[0]:>8.2f}   1% {pct[1]:>8.2f}   "
          f"50% {pct[2]:>7.3f} degC")
    print(f"   -> gate (worst > -5 degC) on the DATA: "
          f"{'PASS' if worst > -5.0 else 'FAIL'}")

# A tolerance that the data itself satisfies, for reference.
allsel = np.concatenate([idx[k] for k in ("train", "valid", "test")])
d_all = np.diff(tc[allsel][:, :GROWTH_END_STEP + 1, :], axis=1)
print(f"\nacross all 185 sims: worst {d_all.min():.2f} degC, "
      f"0.1st percentile {np.percentile(d_all, 0.1):.2f} degC")
print("A gate the ground truth passes needs a tolerance near the data's own "
      "worst drop,\nor should be applied to a spatial/temporal mean rather than "
      "per-sensor per-step.")

# The same check on the cross-sensor MEAN, which is what "the fire is growing"
# actually claims — individual LES thermocouples fluctuate hard.
m_all = tc[allsel][:, :GROWTH_END_STEP + 1, :].mean(axis=2)
dm = np.diff(m_all, axis=1)
print(f"\ncross-sensor MEAN growth diffs: worst {dm.min():.2f} degC, "
      f"0.1st pct {np.percentile(dm, 0.1):.2f} degC, "
      f"{(dm < -5).sum()} of {dm.size} below -5")
