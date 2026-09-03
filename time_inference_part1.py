"""Measure surrogate inference cost on the Part1 test split.

The thesis claims the surrogate replaces a 12-27 hour FDS run with a prediction
"in milliseconds". That claim was never measured. This script measures it, on
the same models and the same held-out simulations the accuracy figures use.

Reported per simulation (one configuration -> 16 thermocouple histories of 181
steps, plus the 5 heat-release channels):

    * cold  : first call, including CUDA context and kernel autotune
    * warm  : median over repeated calls after warm-up
    * batch : whole test split in one forward pass, divided by the case count

    python time_inference_part1.py --model-dir models_part1_kanbal_s48_r1
"""
import argparse
import io
import statistics as st
import sys
import time

import torch

import evaluate_part1 as E
from config_part1 import DEVICE
from data_loader_part1 import build_dataset, prepare_data_splits

if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

REPEATS = 20


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def forward(models, params, time_array):
    for m in models:
        m(params, time_array)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--split", default="test")
    a = ap.parse_args()

    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    models, tc_scaler, hrr_scaler, ckpt = E.load_ensemble(a.model_dir, device)
    params, tc, hrr, mask, meta, names = build_dataset(verbose=False)
    datasets, _, _, info, _ = prepare_data_splits(
        params, tc, hrr, mask, meta, mode=ckpt.get("split_mode"), verbose=False)
    ds = datasets[a.split]

    P = ds.params.to(device)
    T = ds.time_array.to(device)
    n = len(ds)

    # cold: first call on this process
    sync(device); t0 = time.perf_counter()
    forward(models, P[:1], T[:1])
    sync(device); cold = time.perf_counter() - t0

    # warm: single case, repeated
    for _ in range(3):
        forward(models, P[:1], T[:1])
    sync(device)
    singles = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        forward(models, P[:1], T[:1])
        sync(device)
        singles.append(time.perf_counter() - t0)

    # batch: the whole split at once
    forward(models, P, T)
    sync(device)
    batches = []
    for _ in range(max(5, REPEATS // 4)):
        t0 = time.perf_counter()
        forward(models, P, T)
        sync(device)
        batches.append(time.perf_counter() - t0)

    warm = st.median(singles)
    batch_total = st.median(batches)

    print("model dir      : %s" % a.model_dir)
    print("device         : %s" % (torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"))
    print("ensemble       : %d member(s)" % len(models))
    print("test split     : %d simulations" % n)
    print()
    print("cold start (1 case, incl. context)   : %8.1f ms" % (cold * 1e3))
    print("warm, 1 case, median of %2d           : %8.2f ms" % (REPEATS, warm * 1e3))
    print("whole split in one pass              : %8.1f ms  (%d cases)" % (batch_total * 1e3, n))
    print("  -> per simulation, batched         : %8.2f ms" % (batch_total / n * 1e3))
    print()
    fds_hours = 19.5   # midpoint of the 12-27 h per-case range reported in the thesis
    print("FDS reference  : %.1f h per case (midpoint of 12-27 h)" % fds_hours)
    print("speed-up, warm single case           : %8.3g x" % (fds_hours * 3600 / warm))
    print("speed-up, batched                    : %8.3g x" % (fds_hours * 3600 / (batch_total / n)))


if __name__ == "__main__":
    main()
