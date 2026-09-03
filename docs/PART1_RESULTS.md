# Part1 geometry corpus — results record

Corpus: `D:\Bs8414_05052026\Part1\_completed`, 185 usable sims.
Target: 16 external thermocouples + 5-channel `_hrr.csv` budget, 181 steps.
Split: `PART1_SPLIT=hash` → 142 train / 20 valid / 23 test.
Eval: `evaluate_part1.py` (shared, byte-identical across the sensor projects).
Ensembles: 3 members, seeds 42/43/44, paired across projects.

## Runs

| Run | Model | Params | Valid TC R² | Test TC R² | Combined | Test HRR R² | Test TC RMSE |
|---|---|---|---|---|---|---|---|
| `models_part1_r1` | KAN-Attention-LSTM | 850,765 | 0.7643 | 0.8049 | **0.7846** | 0.9408 | 63.08 °C |
| `models_part1_r2_postfix` | KAN-Attention-LSTM | 850,765 | 0.8146 | 0.8497 | **0.8321** | 0.9518 | 55.37 °C |
| `bs8414_MLP_surrogate/models_part1_mlp_r1` | MLP-Attention-LSTM | 851,012 | 0.7303 | 0.7543 | **0.7423** | 0.9475 | 70.79 °C |

Per-member best valid loss (tc + 0.3·hrr, standardised):
- KAN r1: 0.464 / 0.742 / 0.534 — stopped at epoch 288 / 229 / 344
- KAN r2: 0.482 / 0.490 / 0.516 — stopped at epoch 263 / 380 / 208
- MLP r1: 0.552 / 0.671 / 0.675 — stopped at epoch 291 / 242 / 351

`models_part1_r1` and `models_part1_r2_postfix` are the **same code path** — the
trainer fix described below did not alter the KAN's execution (its `LAMBDA_REG` is
non-zero, so the guarded branch always ran). They are therefore two independent
retrains of identical code, and the 0.048 combined-R² spread between them is a
direct measurement of this corpus's retrain variance — consistent with the ~0.05
test-R² swing the 60-sim project documents.

## What is and is not established

- **The KAN reaches test TC R² 0.80–0.85 and test HRR R² 0.94–0.95 on this corpus.**
  Solid: two independent retrains, both physics gates passing.
- **The HRR head works.** R² 0.93–0.95 on total HRR across every run and split. The
  `Q_TOTAL` channel is the exception at R² ≈ 0.12, but it is a residual budget term
  of ~18 kW RMSE against a ~4000 kW fire — numerically near noise, and not a
  meaningful target. Consider dropping it from `HRR_CHANNELS`.
- **KAN vs MLP is NOT established.** Combined R² 0.7846 / 0.8321 (KAN, n=2) versus
  0.7423 (MLP, n=1). The gap to the KAN's *weaker* run is 0.042 — outside the ±0.02
  band, but comparing a 2-run population to a 1-run population is not a result.
  Needs 3 replicate ensembles each before any claim, per the ablation contract in
  `bs8414_MLP_surrogate/CLAUDE.md`.

## Geometry breakdown (KAN r1 test, n per row is small — directional only)

`nocb` (cavity barriers removed) is consistently the hardest geometry:
test R² 0.67 against 0.88–0.90 for the others, and the same ordering appears in
train (0.79 vs 0.85–0.87) and valid (0.63). Plausible mechanism: with no barriers
the cavity flow is unobstructed and the plume path becomes more chaotic, so the
point thermocouples are less predictable. Worth a dedicated look before it goes in
a paper — per-geometry n is 1–5 on test.

## Defects found and fixed during this run

1. **The optimiser step was disabled for any model with `LAMBDA_REG = 0.0`.** A
   refactor to the uniform interface changed `if train:` to
   `if train and LAMBDA_REG:` in `run_epoch`, which pulled `zero_grad/backward/
   step` inside the regulariser guard. The KAN was unaffected (`LAMBDA_REG=2e-3`);
   the MLP, Attention-LSTM V3 and MLP-Samba variants never took a single step and
   reported flat loss with "best @ epoch 0" — which reads exactly like instant
   convergence. The first MLP run is preserved as
   `models_part1_mlp_r1_BROKEN_no_optimiser_step/` with
   `train_part1_run1_BROKEN.log`. Fixed, and `train_part1.py` now asserts the
   weights actually moved after epoch 0 (`[sentinel] |dW| = ...` in every log).

2. **The growth-monotonicity physics gate was mis-calibrated and failed on the
   ground truth.** It demanded no per-sensor step drop beyond −5 °C during
   0–720 s; the FDS data itself has a worst drop of −440 °C with **26 % of all
   growth-phase steps below −5 °C** (`_diag_growth_gate.py`). An LES point
   thermocouple is not monotonic at DT=10 s. The gate now compares the prediction
   against the ground truth on the same cases and asks whether the model is *less*
   physical than the simulation it imitates. Under that gate all runs pass, and the
   surrogate is markedly smoother than the truth (worst drop −91 °C vs −292 °C;
   late-time rise 28.5 vs 93.6 °C/step) — expected for an ensemble mean.

## Next

- 2 more MLP ensembles + 1 more KAN, for a 3-vs-3 population comparison.
- Train the other two sensor variants (`bs8414_surrogate_model` V3 baseline,
  `bs8414_samba_mlp_surrogate`) — both were affected by defect 1 and have not been
  trained on Part1 since the fix.
- `PART1_SPLIT=system` for the generalisation-to-unseen-build-up number.
- Slice surrogates on the extracted Part1 slices.
