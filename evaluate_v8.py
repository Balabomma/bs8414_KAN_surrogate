"""Evaluate a saved v8 KAN ensemble.

v8 uses the same model class as v7 (KANAttentionLSTMv7, ambient clamp) — only
the training loss differs — so it reuses the frozen v6/v7 eval contract verbatim,
just pointed at models_v8.

Usage: python evaluate_v8.py --model-dir models_v8
"""
import argparse
import evaluate_v6
from model_v7 import KANAttentionLSTMv7

evaluate_v6.KANAttentionLSTMv6 = KANAttentionLSTMv7
from evaluate_v6 import evaluate  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="models_v8")
    args = p.parse_args()
    evaluate(args.model_dir)
