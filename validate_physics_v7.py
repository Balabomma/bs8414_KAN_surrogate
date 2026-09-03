"""Physics-sanity validation for the v7 (ambient-clamp) KAN ensemble.

Same checks as validate_physics_v6; only the model class is swapped so the
ambient_scaled buffer is restored. Sub-ambient count should now be zero.

Usage: python validate_physics_v7.py --model-dir models_v7
"""
import argparse
import evaluate_v6
from model_v7 import KANAttentionLSTMv7

evaluate_v6.KANAttentionLSTMv6 = KANAttentionLSTMv7

import validate_physics_v6  # noqa: E402

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="models_v7")
    args = p.parse_args()
    import sys
    sys.argv = ["validate_physics_v6", "--model-dir", args.model_dir]
    validate_physics_v6.main()
