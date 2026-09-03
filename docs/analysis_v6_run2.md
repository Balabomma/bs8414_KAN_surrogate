# analysis_v6_run2 — KAN-Attention-LSTM v6 retrain (2026-07-08)

**Parent:** v6 run1 (2026-07-07, `models_v6_51sim_jul07/`). **One change:** none to code —
retrain of the *same* `train_v6.py` on the **grown dataset** (51 → 63 sims). This is a
data-delta run, not an architecture mutation.

## Config diff vs parent
- Sims: 51 → **63** (12 new FDS runs landed: more Test_1_PE_PIR HRR + LCM mesh coverage).
- Split: 33/10/8 → **41/13/9** (same deterministic hash split, more sims to bucket).
- Everything else identical: 12 candidates → greedy-5 by pooled valid R², 39-feature
  physics-causal input, plume channel, per-cladding anchor jitter, `PhysicsLossV5`,
  852,795 params/member. Selected M12/M4/M8/M6/M2. Runtime 5914 s (~98 min) on the 4090.

## Metrics (evaluate_v6.py, masked denormalised °C)

| Set | R² | RMSE (°C) | MAE (°C) | MAPE(T>100) |
|-----|:--:|:---------:|:--------:|:-----------:|
| Train | 0.9420 | 55.5 | 31.8 | 11.3% |
| **Valid** | **0.8057** | 98.7 | 51.1 | 20.1% |
| **Test** | **0.8921** | 70.4 | 39.7 | 15.3% |

Combined-mean valid+test R² = **0.849** (parent 51-sim: 0.833).
Test per-sensor: avg R² 0.819, 14/24 sensors R²>0.8, 8/24 R²>0.9.

### vs parent (v6 run1, 51-sim / 33-10-8) — NOT a controlled comparison (split + sim count differ)
| | Valid R² | Test R² | Valid MAPE | Test MAPE |
|--|:--:|:--:|:--:|:--:|
| v6 run1 (51) | 0.7707 | 0.8949 | 20.6% | 13.7% |
| v6 run2 (63) | **0.8057** | 0.8921 | 20.1% | **15.3%** |

- **Valid R² crossed 0.8 for the first time** (0.771 → 0.806). Attributable to the 12 new
  sims (better Test_1/LCM mesh-HRR coverage — exactly the lever flagged in kan-v6-campaign),
  **not** to any code change. Not an architecture claim.
- Test R² flat (0.895 → 0.892) — inside the ±0.02 variance band, i.e. unchanged.
- **Test MAPE regressed 13.7% → 15.3%**, now just over the <15% target. Valid MAPE 20.1%
  still fails <15%. So on MAPE, criteria are now *not* met on either set.
- Weakest valid sims: sim0 `Test_1_PE_PIR HRR1333 M010` R²=0.388 (the chaotic ignition-shift
  killer sim, persists), sim7 `Test_5_LCM_PIR HRR2333 M008` R²=0.447 (LCM mesh bifurcation).
- Weakest test sim: `Test_1_PE_PIR HRR1333 M009` R²=0.745 (same killer family).

## Physics sanity (validate_physics_v6.py)
1. **Sub-ambient excursions — FLAG.** Valid 4.7% of points, Test 4.0%, min −69/−75 °C.
   This is the known *unclamped-output* issue (campaign noted −35 °C); magnitude is now
   larger. **Actionable.**
2. Growth-limb monotonicity — **PASS** (drops >2 °C: valid 1.41%, test 1.28%; <2% gate).
3. HRR-vs-meanT correlation — valid r=+0.54 PASS; test r≈0.00 flagged but **confounded, not
   a defect**: ground-truth test corr is also r=+0.052 (9 sims mix 3 claddings). Model
   faithfully reproduces the data. Inconclusive metric on this small mixed set.
4. Late-time plateau |dT/dt| — **PASS** (~0.43 °C/s tail mean).

## Next-mutation hypotheses (v7 candidates)
1. **Hard output clamp at ambient** (`clamp(min=18°C)` on the decoded output, or softplus
   offset) — kills the −70 °C excursions directly instead of via soft penalty. Predicted:
   removes the only surviving physics FLAG, small RMSE improvement in low-T tails; no R²
   regression expected. Cheapest highest-value change. (Symptom→technique: "physics
   violations surviving soft penalties → hard-enforce on the output.")
2. **MAPE-targeted loss reweight** — test MAPE drifted over 15%; the peak/hot-region weight
   or an explicit relative-error term could pull it back under 15% without hurting R².
3. Killer-sim (`Test_1 HRR1333`) still caps valid — improving the anchor predictor
   (±100–160 s error) or awaiting more Test_1 mesh coverage remains the structural lever.

## Verdict
Same-architecture retrain on more data. **Valid R² target (>0.8) now met** (0.806); test
R² target still met (0.892). **MAPE targets not met** on either set. The valid gain is a
data effect, honestly not an architecture win. One physics FLAG (sub-ambient) is real and
becomes the v7 headline change. Checkpoints: `models_v6/` (current, 63-sim),
`models_v6_51sim_jul07/` (preserved parent).
