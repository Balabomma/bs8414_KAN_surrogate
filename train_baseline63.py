"""Retrain the ORIGINAL baseline (train.py, 16-param input, 5-member ensemble)
on the current 63-sim dataset, saving to models_baseline_63sim/ so the May
reference checkpoint in models/ stays untouched.

Purpose: FireSnC2026 R7 Table 5 — baseline and physics-causal-v9 cells must both
come from models trained on the identical 63-sim (41/13/9) dataset.

Usage: python train_baseline63.py
"""
import os
import config

config.MODEL_DIR = os.path.join(config.PROJECT_DIR, "models_baseline_63sim")

import train  # noqa: E402  (binds MODEL_DIR from config at import time)

assert train.MODEL_DIR.endswith("models_baseline_63sim"), train.MODEL_DIR

if __name__ == "__main__":
    train.main()
