# analysis_v8_run1 — KAN v8 = v7 clamp + robust tail-downweighted loss (2026-07-08)

**Parent:** v7 (`models_v7/`, ambient clamp). **One variable:** the data term of the loss.
`PhysicsLossV8` multiplies the peak-weighted MSE by a detached adaptive tail-weight
`tw = 1/(1+(|e|/τ)²)`, τ=1.0 (scaled), so heavy-tail (chaotic-spike) points shed gradient.
Ambient clamp, physics-causal inputs, all other loss terms, features, split, eval identical.
Same 852,795 params. Runtime 8397 s (candidates ran longer — flatter loss surface).
Selected M11/M12/M8/M6/M3.

## Why this change (measured, pre-GPU)
- Oracle experiment refuted the anchor-predictor route: perfect anchors → 0 R² gain,
  killer sim stays R²≈0.37. (See conversation / VARIANCE_RECORD v7 block.)
- Real cap = 1–2 LES-chaotic sims (`Test_1_PE_PIR HRR1333`). Scaled residuals heavy-tailed
  (p50 0.10, p90 0.39, p99 1.23, max 9.2). Peak-weighted MSE gave those unfittable spikes
  both huge error and large weight → they dominated the gradient. τ=1.0 keeps p50–p90 at
  87–99% weight, sheds p99→0.40 and max→0.01 (meanTW 0.95).

## Metrics (evaluate_v8.py = frozen contract, masked denorm °C)

| Set | R² | RMSE | MAE | MAPE(T>100) | sub-ambient |
|-----|:--:|:----:|:---:|:-----------:|:-----------:|
| Train | 0.9282 | 61.7 | 26.7 | 10.4% | 0 |
| Valid | **0.8220** | 94.5 | **44.9** | **19.0%** | 0 |
| Test  | 0.8543 | 81.8 | **37.2** | 15.5% | 0 |

## Head-to-head vs v6 (champion) — same split/data/eval

| Metric | v6 run2 | v8 run1 | Δ |
|--------|:-------:|:-------:|:--:|
| Valid R² (full) | 0.8057 | **0.8220** | **+0.016** |
| Valid R² (excl killer) | 0.8601 | **0.8737** | **+0.014** |
| Valid MAE | 51.1 | **44.9** | **−6.2 °C** |
| Test R² (full) | **0.8921** | 0.8543 | **−0.038** |
| Test R² (excl killer) | **0.9127** | 0.8884 | −0.024 |
| Test MAE | 39.7 | **37.2** | −2.5 °C |
| Physics FLAGs | sub-ambient (−75 °C) | **none** (all pass) | ✓ |
| Combined valid+test R² | **0.849** | 0.838 | −0.011 |

## Interpretation — an L1/robustness win, an L2 test-R² loss
The tail-downweight did exactly what it mechanically must: it traded L2 (squared-error)
fit on the hard tail for better bulk fit.
- **Improved:** valid R² (best-ever 0.822, and up even excluding the killer), **MAE on
  both sets** (valid −6.2 °C, test −2.5 °C), **valid MAPE** (best), and **physics** —
  every real sanity check now PASSes (growth 1.4–1.6%, plateau, 0 sub-ambient; only the
  known-confounded HRR-corr flag remains, ground-truth test corr is also ≈0).
- **Regressed:** test R² −0.038 full, and −0.024 **even excluding the killer sim** — so
  the shed isn't only the killer; downweighting moderately-hard test points raised their
  squared error too. Test is L2-dominated by a few hard cases, so it loses where MAE gains.
- Net **combined R² 0.838 < v6 0.849** → champion stays **v6 run2** on the combined-R²
  contract. Test per-sensor avg R² fell to 0.723 (v6 0.819) — L2 spread up, L1 down.

## Verdict vs the goal ("improve both test and validation R²")
- **Validation R²: improved** (+0.016, top of the variance band, and holds excl-killer).
- **Test R²: not improved — regressed** (−0.038, outside the ±0.02 band). Goal not met on
  test. This is a real mechanism (L2↔L1 tension), not pure draw noise: the test excl-killer
  R² also fell and the direction matches the tail-weight's design.
- v8 is the **best "balanced/robust" model** (MAE + physics + valid R²) but **not** a test-R²
  win. τ=1.0 is too aggressive for the L2 test objective.

## Next step (v9 candidate) — gentler shed
τ=1.0 sheds test's moderately-hard points too. A **τ≈1.6–2.0** would downweight only the
extreme max-tail (keep p95 near full weight), likely retaining most of the valid/MAE gain
while giving test's L2 points back their gradient → recover test R² toward v6. Single
variable (τ), cheap. If test R² is THE objective, this is the lever; if balanced MAE+physics
is acceptable, v8 already wins those.

Checkpoints: `models_v8/` (+outputs). Champion unchanged: `models_v6/`.
Logs: `train_v8_run1.log`, `evaluate_v8_run1.log`, `validate_physics_v8_run1.log`.
