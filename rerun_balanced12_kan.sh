#!/usr/bin/env bash
# Balanced 12-run design for the KAN, matching the MLP ablation control exactly:
# 3 independent retrains at each of 4 base seeds (42, 45, 48, 52).
#
# Purpose: separate the two variance components the reported +/-0.038 conflates.
#   within-seed  - cudnn.benchmark leaves the per-member draw non-deterministic,
#                  so repeated runs at one seed differ. Measured at ~0.018 on
#                  five seed-42 runs.
#   between-seed - seed 42 scored 0.744 against 0.806 for seeds 45/48/52, a gap
#                  of 0.062, i.e. 3.4x the within-seed spread. The eight-run pool
#                  is seed-42-heavy (5 of 8), so its mean is composition-dependent.
#
# All 12 runs are FRESH. The existing eight are not reused: their values are
# already known, so selecting three seed-42 runs from five would be selection on
# outcome. They remain available as an independent replication check.
#
# Design is balanced (n=3 per cell) so the within/between decomposition is a
# plain one-way ANOVA with equal cells - no weighting decisions to defend.
set -u
cd "$(dirname "$0")"
PY=./venv/Scripts/python.exe

run () {                                  # $1 = seed, $2 = replicate index
  d="models_part1_kanbal_s$1_r$2"
  if [ -d "$d" ]; then echo "  SKIP $d (exists)"; return; fi
  echo "  === seed $1  replicate $2  -> $d ==="
  $PY -u train_part1.py --model-dir "$d" --members 3 --seed "$1" \
      > "train_part1_kanbal_s$1_r$2.log" 2> "train_part1_kanbal_s$1_r$2.err.log"
  $PY -u evaluate_part1.py --model-dir "$d" \
      > "evaluate_part1_kanbal_s$1_r$2.log" 2> "evaluate_part1_kanbal_s$1_r$2.err.log"
  echo "      done $(date +%H:%M)"
}

for s in 42 45 48 52; do
  for r in 1 2 3; do
    run "$s" "$r"
  done
done

echo "  BALANCED 12 COMPLETE $(date +%H:%M)"
