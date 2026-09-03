"""Best-of-N replicates for the clean-60 v9 recipe (tau=1.8), for mean+/-spread.

Replicate 1 = existing models_v9 (v9 run2). This driver runs replicates 2 and 3
into SEPARATE named dirs so nothing overwrites (VARIANCE_RECORD lesson: sequential
retrains once clobbered models/ and were lost). Same code, same clean-60 split;
cudnn.benchmark non-determinism gives a different ensemble draw each run.

Run unbuffered:  python -u bestofn_v9_driver.py
"""
import os
import time
from config import PROJECT_DIR
from train_v8 import main

TAU = 1.8
REPLICATES = ["models_v9_r2", "models_v9_r3"]

if __name__ == "__main__":
    t0 = time.time()
    for name in REPLICATES:
        d = os.path.join(PROJECT_DIR, name)
        print(f"\n########## RETRAIN START: {name}  (t={time.time()-t0:.0f}s) ##########",
              flush=True)
        main(model_dir=d, tau=TAU)
        print(f"########## RETRAIN DONE: {name}  (t={time.time()-t0:.0f}s) ##########",
              flush=True)
    print(f"\n########## ALL REPLICATES DONE  (total {time.time()-t0:.0f}s) ##########",
          flush=True)
