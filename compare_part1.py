"""Aggregate Part1 replicate ensembles into a population comparison.

Reads the `evaluation_part1.json` that `evaluate_part1.py` writes into each
`models_part1*/` directory across the four thermocouple projects and reports
mean +/- spread per model, rather than single runs.

    python compare_part1.py                    # table + verdicts
    python compare_part1.py --log              # also append to comparison_logs/

Why populations: two retrains of byte-identical KAN code on this corpus differ by
0.048 combined R2. A single-run delta smaller than that is weather. The verdict
column applies the project's standing rule — a difference inside +/-0.02 R2 is
reported inconclusive, in either direction, and a difference is only called when
the two populations do not overlap.

`*_BROKEN*` directories are skipped by name; they hold the run where the optimiser
step was disabled and are kept only as evidence.
"""
import argparse
import glob
import json
import os
import statistics
from datetime import datetime, timezone

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(PROJECT_DIR)

# display name -> (project dir, glob for its Part1 run directories)
MODELS = {
    "KAN-Attention-LSTM": ("bs8414_KAN_surrogate", "models_part1_r*"),
    "MLP-Attention-LSTM": ("bs8414_MLP_surrogate", "models_part1_mlp_r*"),
    "Attention-LSTM V3": ("bs8414_surrogate_model", "models_part1_v3_r*"),
    "MLP-Samba": ("bs8414_samba_mlp_surrogate", "models_part1_samba_r*"),
}

BASELINE = "KAN-Attention-LSTM"
BAND = 0.02  # deltas inside this are inconclusive, in either direction


def collect(project, pattern):
    """One row per replicate ensemble that has been evaluated."""
    runs = []
    for d in sorted(glob.glob(os.path.join(ROOT, project, pattern))):
        base = os.path.basename(d)
        # BROKEN   = the run where the optimiser step was disabled.
        # DUPLICATE = a same-seed repeat that came out bit-identical (the V3
        #             architecture is deterministic on this GPU), so it carries
        #             no variance information and must not inflate n.
        if "BROKEN" in base.upper() or "DUPLICATE" in base.upper():
            continue
        path = os.path.join(d, "evaluation_part1.json")
        if not os.path.isfile(path):
            runs.append({"dir": base, "evaluated": False})
            continue
        with open(path) as f:
            ev = json.load(f)
        if not ev.get("valid") or not ev.get("test"):
            continue

        seeds = []
        summary = os.path.join(d, "run_summary.json")
        if os.path.isfile(summary):
            with open(summary) as f:
                seeds = [m.get("seed") for m in json.load(f).get("members", [])]

        runs.append({
            "dir": base,
            "seeds": seeds,
            "evaluated": True,
            "valid_r2": ev["valid"]["tc_r2"],
            "test_r2": ev["test"]["tc_r2"],
            "combined": ev.get("combined_tc_r2",
                               0.5 * (ev["valid"]["tc_r2"] + ev["test"]["tc_r2"])),
            "test_rmse": ev["test"]["tc_rmse"],
            "test_hrr_r2": ev["test"]["hrr"]["HRR"]["r2"],
            "physics_pass": all(
                ev[s]["physics"].get(k, True)
                for s in ("valid", "test")
                for k in ("sub_ambient_pass", "growth_monotonic_pass",
                          "late_plateau_pass")),
        })
    return runs


def spread(values):
    """mean, and half-range as the spread (n is 3 — sd is not meaningful)."""
    if not values:
        return None, None
    mean = statistics.fmean(values)
    return mean, (max(values) - min(values)) / 2.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", action="store_true",
                    help="append the table to bs8414_surrogate_model/comparison_logs/")
    args = ap.parse_args()

    print("=" * 96)
    print("  Part1 geometry corpus — population comparison")
    print("  185 sims, hash split 142/20/23, 16 external TCs + 5 HRR channels")
    print("=" * 96)

    populations = {}
    pending = []

    for name, (project, pattern) in MODELS.items():
        runs = collect(project, pattern)
        done = [r for r in runs if r["evaluated"]]
        pending += [f"{name}: {r['dir']}" for r in runs if not r["evaluated"]]
        if not done:
            print(f"\n  {name:<22} no evaluated runs yet")
            continue
        populations[name] = done

        seed_sets = {tuple(r.get("seeds") or []) for r in done}
        print(f"\n  {name}  ({project}, n={len(done)}, "
              f"{len(seed_sets)} distinct seed set(s))")
        print(f"    {'run':<26}{'seeds':>12}{'valid R2':>10}{'test R2':>10}"
              f"{'combined':>10}{'test RMSE':>11}{'HRR R2':>9}  physics")
        for r in done:
            s = r.get("seeds") or []
            stag = f"{s[0]}-{s[-1]}" if s else "?"
            print(f"    {r['dir']:<26}{stag:>12}"
                  f"{r['valid_r2']:>10.4f}{r['test_r2']:>10.4f}"
                  f"{r['combined']:>10.4f}{r['test_rmse']:>10.2f}C"
                  f"{r['test_hrr_r2']:>9.4f}  "
                  f"{'PASS' if r['physics_pass'] else 'FAIL'}")
        if len(seed_sets) == 1 and len(done) > 1:
            print(f"    NOTE: every run used the same seeds — this population "
                  f"measures GPU non-determinism only.")

        for key, label in (("valid_r2", "valid R2"), ("test_r2", "test R2"),
                           ("combined", "combined"), ("test_hrr_r2", "HRR R2")):
            m, s = spread([r[key] for r in done])
            print(f"    {'mean +/- half-range':<26}{label:>10}: "
                  f"{m:.4f} +/- {s:.4f}")

    if pending:
        print(f"\n  not yet evaluated ({len(pending)}):")
        for p in pending:
            print(f"    - {p}")
        print("    run: python evaluate_part1.py --model-dir <dir>  in that project")

    # ── verdicts vs the baseline ──────────────────────────────────────────
    if BASELINE in populations and len(populations) > 1:
        base = [r["combined"] for r in populations[BASELINE]]
        b_mean, b_spread = spread(base)
        print(f"\n  verdicts on combined valid+test TC R2, vs {BASELINE} "
              f"({b_mean:.4f} +/- {b_spread:.4f}, n={len(base)})")
        print(f"    rule: |delta| < {BAND} -> inconclusive; populations that "
              f"overlap -> inconclusive")

        for name, runs in populations.items():
            if name == BASELINE:
                continue
            vals = [r["combined"] for r in runs]
            m, s = spread(vals)
            delta = m - b_mean
            overlap = (min(vals) <= max(base)) and (min(base) <= max(vals))

            if abs(delta) < BAND:
                verdict = f"INCONCLUSIVE (|delta| {abs(delta):.4f} < {BAND})"
            elif overlap:
                verdict = (f"INCONCLUSIVE (delta {delta:+.4f} but the "
                           f"populations overlap)")
            elif len(vals) < 3 or len(base) < 3:
                verdict = (f"SUGGESTIVE only (delta {delta:+.4f}; needs n>=3 "
                           f"both sides, have {len(vals)} vs {len(base)})")
            else:
                verdict = (f"{'better' if delta > 0 else 'WORSE'} by "
                           f"{abs(delta):.4f} R2, populations disjoint")
            print(f"    {name:<22} {m:.4f} +/- {s:.4f}  (n={len(vals)})  -> {verdict}")

    if args.log and populations:
        log_dir = os.path.join(ROOT, "bs8414_surrogate_model", "comparison_logs")
        os.makedirs(log_dir, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%MZ")
        path = os.path.join(log_dir, f"part1_population_{stamp}.json")
        with open(path, "w") as f:
            json.dump({"corpus": "Part1/_completed", "split": "hash",
                       "n_sims": 185, "populations": populations}, f, indent=1)
        print(f"\n  appended: {path}")


if __name__ == "__main__":
    main()
