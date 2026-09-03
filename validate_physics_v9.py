"""Physics-sanity validation for the v9 KAN ensemble (same model class as v7/v8).

Usage: python validate_physics_v9.py --model-dir models_v9
"""
import argparse
import sys
import evaluate_v6
from model_v7 import KANAttentionLSTMv7

evaluate_v6.KANAttentionLSTMv6 = KANAttentionLSTMv7
import validate_physics_v6  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="models_v9")
    args = p.parse_args()
    sys.argv = ["validate_physics_v6", "--model-dir", args.model_dir]
    validate_physics_v6.main()
