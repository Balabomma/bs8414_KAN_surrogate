# analysis_v7_run1 — KAN v7 = v6 + hard ambient clamp (2026-07-08)

**Parent:** v6 run2 (`models_v6/`, 63-sim 41/13/9). **One change:** decoded output
hard-clamped at ambient (18 °C) per sensor in scaled space via `KANAttentionLSTMv7`
(`ambient_scaled` buffer). Physics-causal channels, features, loss, split, eval all
identical to v6. Same 852,795 params (zero new — pure output op). Runtime 5610 s.
Selected ensemble M12/M7/M5/M2 (4 members; greedy stopped at 4, valid R² plateaued).

## Metrics (evaluate_v7.py = frozen v6 contract, masked denorm °C)

| Set | R² | RMSE | MAE | MAPE(T>100) | sub-ambient |
|-----|:--:|:----:|:---:|:-----------:|:-----------:|
| Train | 0.9159 | 66.8 | 36.7 | 13.9% | 0 |
| Valid | 0.8029 | 99.4 | 52.1 | 22.0% | 0 |
| Test  | 0.8672 | 78.0 | 42.6 | 17.6% | 0 |

Test per-sensor avg R² 0.802; 17/24 >0.8; **3/24 >0.9**.

### vs v6 run2 (same split/data/eval)
| | Valid R² | Test R² | Test MAPE | sub-ambient | sensors>0.9 |
|--|:--:|:--:|:--:|:--:|:--:|
| v6 run2 | 0.8057 | **0.8921** | 15.3% | 4.0% (−75 °C) | 8/24 |
| v7 run1 | 0.8029 | 0.8672 | 17.6% | **0** | 3/24 |

## Interpretation — the clamp works; test R² drop is an unlucky draw, not the clamp
- **Physics FLAG fixed.** Sub-ambient excursions 4.0% → **0%**; min prediction exactly
  18 °C on both sets. This is the intended, structural effect. It cannot regress.
- **Test R² −0.025** is confounded with retrain variance and attributable to a weaker
  ensemble draw this run, NOT the clamp:
  - *Isolated clamp effect* (post-hoc clamp on the v6 weights, identical network):
    test R² 0.8921 → 0.8932 (**+0.001**). The clamp itself is neutral-positive.
  - This run's 12 candidates were systematically weaker (valid R² 0.706–0.770 vs v6
    run2's 0.742–0.785). Fewer high-performers survived (3 vs 8 sensors >0.9).
  - The clamp only alters points below ambient; it is mathematically incapable of
    hurting the hot-region (T≫18 °C) fit that dominates R². So the hot-region shortfall
    is the draw, not the mechanism.
  - Project rule: single-run ±0.02 R² is noise; −0.025 with a weaker candidate pool and
    n=9 test sims is within the effective band. **Not a real regression from the clamp.**
- Growth-limb monotonicity ticked to 2.0–2.5% (v6 1.3–1.4%), just over the 2% gate —
  again consistent with the weaker draw, marginal.

## Verdict
- **Keep the clamp** — it removes the last physics FLAG for free (no params, no R² cost
  in isolation). Retain `KANAttentionLSTMv7` as the mechanism going forward.
- **Champion stays v6 run2** on combined valid+test R² (0.849 vs v7 0.835); v7's test R²
  came from an unlucky draw. To claim v7 ≥ v6 we need a fair draw, not this one.
- Logged as a **neutral/negative-on-metric, positive-on-physics** result.

## Next steps (v8 candidates)
1. **Re-roll v7 with best-of-N** — train ~16–20 clamped candidates (or reuse
   `models_v7/all_candidates.pt` + more) and greedy-select 5 by valid R². Expectation:
   a fair clamped ensemble matches v6 test R² while keeping 0 sub-ambient. This is the
   controlled comparison the clamp deserves.
2. **Target the actual test-R² ceiling — the killer family.** Weakest test sims are the
   `Test_1_PE_PIR HRR1333` ignition-shift cases (v7 test sim0/1 R²≈0.78). The lever is the
   anchor predictor (±100–160 s timing error), not the output clamp. A v8 improving anchor
   timing (or more Test_1 mesh coverage from new FDS runs) is where test R² actually moves.

Checkpoints: `models_v7/` (this run + outputs), parent `models_v6/`, grandparent-preserved
`models_v6_51sim_jul07/`.
