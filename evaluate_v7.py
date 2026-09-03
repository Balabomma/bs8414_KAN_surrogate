"""Evaluate a saved v7 KAN ensemble (v6 eval contract + ambient-clamp model).

Reuses evaluate_v6's frozen evaluation logic verbatim; only the model class is
swapped to KANAttentionLSTMv7 so the ambient_scaled buffer is restored from the
checkpoint. Keeping the same eval script keeps v6 and v7 directly comparable.

Usage: python evaluate_v7.py --model-dir models_v7
"""
import argparse
import evaluate_v6
from model_v7 import KANAttentionLSTMv7

# swap the class referenced inside evaluate_v6.load_ensemble_v6
evaluate_v6.KANAttentionLSTMv6 = KANAttentionLSTMv7

from evaluate_v6 import evaluate  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="models_v7")
    args = p.parse_args()
    evaluate(args.model_dir)
