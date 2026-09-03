"""Evaluate a saved v9 KAN ensemble (same model class as v7/v8: ambient clamp).

Usage: python evaluate_v9.py --model-dir models_v9
"""
import argparse
import evaluate_v6
from model_v7 import KANAttentionLSTMv7

evaluate_v6.KANAttentionLSTMv6 = KANAttentionLSTMv7
from evaluate_v6 import evaluate  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="models_v9")
    args = p.parse_args()
    evaluate(args.model_dir)
