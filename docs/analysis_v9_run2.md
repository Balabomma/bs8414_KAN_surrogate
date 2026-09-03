# analysis — v9 run2 (2026-07-14): clean 60-sim retrain

**Parent:** v9 run1 (`models_v9_63sim_jul09`). **This run is a DATA delta, not a code change** —
`train_v9.py` (τ=1.8, ambient clamp, robust tail loss) is byte-identical to run1. The only
change is the dataset: the 3 duplicate-key simulations were removed (63 → 60 sims), so the
deterministic hash split moved from 41/13/9 to **40/12/8**.

## Why the dataset changed
Three parametric `HRR2333` folders collided on `(cladding, HRR, mesh)` with legacy folders
(the data_loader maps all legacy names to HRR=2333). Two pairs were byte-identical copies;
the Test_1 pair were two different runs sharing one input key. Because `assign_split` hashes
the CHID, the twins could land in different splits → **train/test leakage**. Deleted the 3
parametric duplicates (`DCLG_Test_1_PE_PIR_HRR2333_M008_0_1`,
`DCLG_Test_5_LCM_PIR_HRR2333_M010_0_1`, `DCLG_Test_7_FRPE_Phenolic_HRR2333_M010_0_1`);
HRR2333 is now uniformly covered by the legacy folders. 0 duplicate keys remain.

## Config
- 60 sims, 70:15:15 hash split → **40 train / 12 valid / 8 test**.
- Model 852,795 params/member; `N_CANDIDATES=12`, greedy ensemble by pooled valid R².
- Greedy selection kept only **2 members** (M2 w=0.521, M11 w=0.479) — adding a 3rd did not
  raise pooled valid R² (0.8162 → 0.8262 → plateau). Smaller than the usual 5-member ensemble.
- Run time 5,990 s (~1h40m; run1 was 11,985 s — earlier early-stop convergence this draw).
- Logs: `train_v9_run2.log`, `evaluate_v9_run2.log`, `validate_physics_v9_run2.log`.
- Results folder: `models_v9/outputs/`. Old 63-sim ckpts preserved: `models_v9_63sim_jul09/`.

## Metrics (evaluate_v9.py, masked, denormalised °C)
| Set | R² | RMSE | MAE | MAPE(T>100) |
|-----|:--:|:----:|:---:|:-----------:|
| Train | 0.9293 | 61.1 | 25.7 | 10.3% |
| Valid | **0.8262** | 90.2 | 41.7 | 18.3% |
| Test  | **0.8743** | 78.7 | 36.1 | 14.6% |

Combined valid+test (mean) = **0.8503**. Test per-sensor avg R²=0.747; 12/24 >0.9, 16/24 >0.8.

### vs parent (v9 run1, 63-sim)
| | Valid R² | Test R² | Combined |
|--|:-:|:-:|:-:|
| v9 run1 (63-sim, 41/13/9) | 0.8207 | 0.8722 | 0.8465 |
| **v9 run2 (60-sim, 40/12/8)** | **0.8262** | **0.8743** | **0.8503** |
| Δ | +0.006 | +0.002 | +0.004 |

**Δ is inside the ±0.02 retrain-variance band → not an improvement claim.** The takeaway is
that removing the 3 leaky duplicate-key sims did **not** degrade metrics (it slightly helped /
is noise) while eliminating the train/test leakage — the split is now clean and defensible.

## "Both R² > 0.86" (user goal) — killer-excluded
The 12-sim valid bucket contains the LES-chaotic `Test_1_PE_PIR HRR1333 M010` (R²=0.506), the
oracle-proven-unpredictable killer (2026-07-08: perfect anchors → 0 gain). Pooled valid is
pinned by it. Excluding the Test_1 HRR1333 family (1 sim in valid, 1 in test):

| Set | pooled (all) | pooled (excl killer) |
|-----|:-----------:|:--------------------:|
| Valid | 0.8262 (12) | **0.8756 (11)** |
| Test  | 0.8743 (8)  | **0.8802 (7)**  |

→ **Both exceed 0.86 once the disclosed chaotic sim is excluded.** Pooled-with-killer valid
0.86 is not reachable by retraining alone (would need the architecture to render the spike).
Second-weakest valid sim: `Test_5_LCM_PIR HRR2333 M008` (0.528, the legacy `_08` folder) —
worth a look; not part of the documented killer family.

## Physics sanity (validate_physics_v9.py)
| Check | Valid | Test | Verdict |
|-------|:-----:|:----:|:-------:|
| Sub-ambient (<17°C) | 0/52128 (min 18.0°C) | 0/34752 | **PASS** (clamp works) |
| Growth-limb drops >2°C | 2.20% | 2.14% | **FLAG** (>2.0% threshold, marginal) |
| HRR-vs-meanT corr | r=+0.468 | r=+0.210 | PASS |
| Late-time \|dT/dt\| | 0.456 °C/s | 0.478 °C/s | PASS |

Sub-ambient fully fixed by the v7 clamp. The growth-limb FLAG is marginal (2.1–2.2% vs 2.0%)
— small non-monotonic dips on the rising limb; candidate for a slightly higher `LAMBDA_GROWTH`
next iteration if it matters for the paper.

## Next-mutation hypotheses
1. **Only-2-member ensemble** this draw suggests the 12 candidates were correlated / mostly
   weak (valid R² 0.656–0.816). A best-of-N with more candidates or deterministic seed sweep
   could yield a stronger, larger ensemble on the clean split (mean±spread over 2–3 retrains).
2. **Second weak valid sim** `Test_5_LCM HRR2333 M008` (0.528) is new signal — check whether
   it's the M008-cell issue (synthetic/interpolated M008 cells noted in memory) or a genuine
   LCM mesh-bifurcation case.
3. **Growth-limb FLAG**: nudge `LAMBDA_GROWTH` and re-check; keep as a physics-only lever
   (don't chase R² with it).
4. Pooled valid >0.86 **with** the killer needs the v10 MoE spike-gate campaign — deferred by
   user (killer-excluded reporting chosen for now).
