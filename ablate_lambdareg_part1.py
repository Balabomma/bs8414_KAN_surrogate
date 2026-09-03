"""Ablation: train the Part1 KAN with the spline regulariser switched off.

Why this exists. The parity harness holds the data contract, loader, trainer,
evaluation script, optimiser, schedule and ensemble protocol identical between
the KAN and MLP arms, and the thesis described the model definition as the only
free variable. It is not. `model_part1.LAMBDA_REG` is 2e-3 in the KAN project
and 0.0 in the MLP project, so the KAN objective carries a spline L2 + knot
roughness penalty that the MLP objective has no counterpart for. The measured
KAN advantage is therefore attributable to the edge-activation architecture OR
to that extra penalty, and the two have never been separated.

This driver runs the KAN arm with LAMBDA_REG forced to zero, everything else
untouched, at the same seed bases as the main comparison. Reading:

    KAN(reg=0) ~ KAN(reg=2e-3)   the penalty is not doing the work -> the
                                 architecture comparison stands as reported
    KAN(reg=0) ~ MLP             the penalty was the whole effect -> the
                                 architectural claim must be withdrawn

Nothing shared is modified: LAMBDA_REG is rebound on the already-imported
train_part1 module, which is where `run_epoch` reads it from.

Usage:
    python -u ablate_lambdareg_part1.py > ablate_lambdareg.log 2> ablate_lambdareg.err.log
"""
import os
import subprocess
import sys

import train_part1

# Checked once, at import, while the value is still the project's own.
assert train_part1.LAMBDA_REG != 0.0, "expected a non-zero KAN regulariser"

SEEDS = [45, 48]   # seed 42 completed in the first sweep
PROJECT = os.path.dirname(os.path.abspath(__file__))


def run_one(seed):
    model_dir = f"models_part1_kan_noreg_seed{seed}"
    out = os.path.join(PROJECT, model_dir)
    os.makedirs(out, exist_ok=True)

    # The one intervention. run_epoch closes over train_part1's module global,
    # so rebinding it here is sufficient and touches no shared file. The
    # non-zero check belongs at import, not here: after the first seed the
    # value is already zero and asserting per-call kills the run mid-sweep.
    train_part1.LAMBDA_REG = 0.0

    print(f"\n{'=' * 70}\n  KAN ablation, LAMBDA_REG = 0, seed base {seed}\n"
          f"  -> {model_dir}\n{'=' * 70}", flush=True)

    sys.argv = ["train_part1.py", "--members", "3", "--seed", str(seed),
                "--model-dir", model_dir, "--force"]
    train_part1.main()

    subprocess.run([sys.executable, "evaluate_part1.py", "--model-dir", model_dir],
                   cwd=PROJECT, check=False)


if __name__ == "__main__":
    for s in SEEDS:
        run_one(s)
    print("\n  ablation complete", flush=True)
