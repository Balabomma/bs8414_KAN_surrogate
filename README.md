# bs8414_KAN_surrogate — KAN-Attention-LSTM thermocouple surrogate

Deep-learning surrogate for the **BS 8414-1** large-scale facade fire test:
predicts thermocouple time series (and, on the Part1 corpus, the global energy
budget) directly from a build-up description, with FDS as the ground truth.

The architectural idea under test is **KAN** — every fully-connected layer in the
parameter encoder and the sensor decoders is a `KANLinear`: learnable B-spline
activations *on the edges* (RBF basis, 8 knots, tanh-squashed inputs, SiLU
residual, per-layer LayerNorm) instead of a fixed activation on the nodes. The
conv / LSTM / attention temporal backbone is unchanged, so the comparison against
`bs8414_MLP_surrogate` isolates the edge block. This project is also the origin
of the `KANLinear` block vendored by
`bs8414_fundiff_kan_surrogate` and `bs8414_physicsnemo_kan_surrogate`.

**Two corpora live side by side and must never be mixed** — different targets,
different splits, different files.

| | 60-sim corpus | Part1 geometry corpus |
|---|---|---|
| Files | `config.py`, `data_loader.py`, `model*.py`, `train*.py`, `evaluate_*.py` | the `*_part1.py` set |
| Data | `data/training_data` (local) | `D:\Bs8414_05052026\Part1\_completed` |
| Design axes | cladding(4) × HRR(5) × mesh(3) | cladding(12) × insulation(5) × **geometry(8)** |
| Target | **24 TCs** × 181 steps, 3 groups | **16 external TCs** × 181 steps, 2 groups **+ 5-channel HRR budget** |
| Split | 70/15/15 CHID-hash → **40 / 12 / 8** on the clean 60 | hash → **141 / 20 / 23** of 184 usable |
| Checkpoints | `models/`, `models_v*`, `models_70_15_15*`, `models_80_10_10*` | `models_part1_*` |
| Streamlit | — | `app_part1.py` |

---

## The models

### Part1 — `model_part1.py` (current work)

`KANAttentionLSTM` re-conditioned for the geometry corpus. **850,765 params per
member**, `MODEL_NAME = "KAN-Attention-LSTM (Part1)"`, `LAMBDA_REG = 2e-3`.
`KANLinear` is *imported* from `model.py` rather than re-declared, so the block
under test is byte-identical to the one the 60-sim results were produced with.

Three changes, all forced by the corpus:

1. **Conditioning.** HRR and mesh are constants in Part1 and carried no signal,
   so they are gone. In their place: three categorical axes with their own
   embeddings — cladding (12), insulation (5), geometry (8). **Geometry is a
   single 8-way embedding over the observed flag combinations** (bit 0 `noair`,
   bit 1 `nogap`, bit 2 `nocb`; id 0 = baseline), not three booleans, so the
   model can learn an arbitrary interaction between removing the cavity, the
   gaps and the barriers. The cost, and it constrains what may be asked
   afterwards: **a combination absent from training has no embedding row and
   cannot be predicted at all.**
2. **Two thermocouple groups, not three.** The Insulation LV2 decoder is gone
   with the channels it predicted — 169 of 186 decks instrument the external
   face only, so training that head on 9 % of the corpus would be a fiction.
   This costs the BR 135 **internal** criterion entirely; the **external**
   criterion is unaffected.
3. **An HRR head.** A second decoder predicts `HRR, Q_RADI, Q_CONV, Q_COND,
   Q_TOTAL` from the same temporal features. The burner ramp is identical in
   every deck, so what this head learns is the cladding/insulation combustion
   contribution — the same physics that drives the thermocouples, which is why
   it shares the backbone. Weighted by `LAMBDA_HRR = 0.3` (env-overridable).

Physics is enforced **on the output**, not penalised in the loss (soft penalties
were not holding on the 60-sim corpus): thermocouples clamped at ambient 18 °C,
total HRR clamped at zero.

```powershell
python model_part1.py     # parameter count + forward-pass check
```

### 60-sim — the v1 → v9 lineage

Each version is a full `model_vN.py` + `train_vN.py` pair, kept rather than
overwritten, so any number in the record can be reproduced by the code that
produced it.

| Version | Model file | What it added |
|---|---|---|
| v1 | `model.py` | KAN-Attention-LSTM: cladding embedding + KAN material encoder → time-encoded BiLSTM → multi-head attention → 3 grouped decoders with input→output skip |
| v2 | `train_v2.py` | recipe change only — targeted at raising validation R² |
| v3 | `model_v3.py` | ignition-anchor time channels |
| v4 | `model_v4.py` | extended FDS input vector (35 features) + anchor channels |
| v5 | `model_v5.py` | + causal physics driver channels |
| v6 | `model_v6.py` | + plume-theory causal channel |
| v7 | `model_v7.py` | + hard ambient output clamp |
| v8 | `train_v8.py` | + robust tail-downweighted data loss |
| **v9** | `train_v9.py` | v8 with a gentler tail-downweight (τ 1.0 → **1.8**) — **the champion recipe** |

The v9 recipe (the one `bs8414_MLP_surrogate` transplants verbatim): 39-feature
physics-causal input vector; TimeEncoding → MultiScaleConv(3/9/27) → 2-layer
BiLSTM(96) → 4-head self-attention → 3 grouped decoders + skip; peak-weighted MSE
× tail-downweight plus init / smooth / relative / growth / decay / energy terms
and `LAMBDA_KAN_REG · kan_regularization(model)`; AdamW 2e-3 / wd 2e-4, cosine
warm restarts, EMA 0.999, clip 1.0, ≤1500 epochs, patience 200; **12 candidates →
greedy ensemble ≤5** by pooled valid R², inverse-val-loss weights.

`train_baseline63.py` retrains the *original* baseline (16-param input, 5-member
ensemble) as the reference point for the whole campaign.

---

## Training

Always from this directory, in **this project's own venv**, on the NVIDIA GPU.

```powershell
cd D:\VS_projects\bs8414_KAN_surrogate
.\venv\Scripts\activate
nvidia-smi                             # confirm the 4090 is free before launching
```

### Part1

```powershell
python verify_parity_part1.py                                   # shared layer identical
python -u train_part1.py --members 3 --seed 48 --model-dir models_part1_r5 `
       > train_part1_r5.log 2> train_part1_r5.err.log
python evaluate_part1.py --model-dir models_part1_r5
```

`train_part1.py` options:

| Flag | Default | Meaning |
|---|---|---|
| `--model-dir` | `models_part1` | output directory — **must not already hold a run** |
| `--members` | 1 | ensemble members; the standard protocol is **3**, seeded `seed + m_idx` |
| `--seed` | 42 | base seed; the balanced design uses 42 / 45 / 48 / 52 / 55 / 61 |
| `--split` | `hash` | `hash` or `system` — **different experiments**, not comparable |
| `--epochs` | 500 | max epochs (early stopping, patience 60) |
| `--force` | off | overwrite a directory that already holds checkpoints |

**The refusal to overwrite is deliberate** — three retrains were lost to silent
overwrites on the 60-sim corpus. Pass `--force` only on purpose. Every log also
carries a `[sentinel] |dW| = ...` line after epoch 0, proving the weights
actually moved; that check exists because a refactor once pulled
`zero_grad/backward/step` inside a regulariser guard and every `LAMBDA_REG = 0`
variant trained for zero steps while reporting "best @ epoch 0".

Roughly **2.5 min per member** on the 4090, so a 3-member ensemble is ~8 min.

Environment knobs: `PART1_SPLIT`, `PART1_LAMBDA_HRR` (default 0.3),
`PART1_LAMBDA_CLOSURE` and `PART1_LAMBDA_GEOM` (default 0.0 — the physics
ablations), `PART1_SIMS_DIR`.

### 60-sim

```powershell
python -u train_v9.py                        > train_v9_run3.log 2> train_v9_run3.err.log
python evaluate_v9.py        --model-dir models_v9
python validate_physics_v9.py --model-dir models_v9
python killer_excluded_v9.py  --model-dir models_v9
python -u bestofn_v9_driver.py               # replicate population, mean ± spread
python br135_classification_v9.py            # BR 135 external screen vs FDS truth
```

Earlier versions follow the same pattern (`train_v6.py` → `evaluate_v6.py` →
`validate_physics_v6.py`).

### Reading the run directories

Naming is systematic; the log beside each directory is its provenance.

| Pattern | What it is |
|---|---|
| `models/` | 60-sim production — `models_70_15_15` restored, SHA-verified |
| `models_v9`, `models_v9_r2`, `models_v9_r3` | the v9 replicate population (champion = `models_v9`) |
| `models_70_15_15*`, `models_80_10_10` | split-ratio study; the 80/10/10 run is **excluded from ranking** (different sample sets) |
| `models_baseline_63sim`, `models_v6_51sim_jul07`, `models_v9_63sim_jul09` | corpus-vintage runs — not comparable to clean-60 numbers |
| `models_part1_r1` … `r4_seed48`, `r4_s52` | Part1 main sequence |
| `models_part1_kanbal_s{42,45,48,52,55,61}_r{1,2,3}` | the **balanced 12+ design**: 3 independent retrains at each of several base seeds, to separate within-seed (cudnn non-determinism, ~0.018) from between-seed variance (~0.062) |
| `models_part1_kan_noreg_seed*`, `models_part1_noreg_seed42` | spline-regulariser ablation (`ablate_lambdareg_part1.py`) |
| `models_part1_closure_w01_s48`, `_w05_s48` | energy-closure penalty ablation (`PART1_LAMBDA_CLOSURE`) |
| `models_part1_sys_kan_seed*` | `PART1_SPLIT=system` — the unseen-build-up protocol, **not comparable** to hash runs |
| `models_part1_184_r1`, `*_c184`, `models_part1_final_r1` | runs on the corrected 184-configuration corpus |

`PART1_RESULTS.md` is the results record and `VARIANCE_RECORD.md` the retrain-variance
record; both quote per-member best-valid-loss and stop-epoch numbers straight out
of the logs, which is why the logs stay in version control.

---

## Evaluation

```powershell
python evaluate_part1.py    --model-dir models_part1_r4_seed48
python metrics_full_part1.py --model-dir models_part1_r4_seed48 --split test
python compare_part1.py                      # replicate ensembles -> population comparison
python compare_splits.py                     # hash vs system, side by side
python time_inference_part1.py               # inference cost on the Part1 test split
python dump_per_sensor_ext.py                # per-sensor test metrics
python dump_ts_part1.py                      # per-case time series for the test split
python explain_part1.py                      # SHAP attribution (Cremades et al. 2025)
python causal_part1.py                       # interventional / causal explainability
```

`evaluate_part1.py` is the **frozen evaluation contract** — fixed before any
candidate model existed and shared byte-identical across every Part1 sensor
project. A candidate that needs different scoring is a different experiment, not
a comparable one. It reports, per split:

- pooled and per-group R² / RMSE on the 16 external thermocouples, in °C;
- R² / RMSE per HRR channel, in kW;
- a **per-geometry breakdown** — the point of this corpus is whether removing the
  cavity, the gaps or the barriers is predictable, so a pooled number that hides
  a failure on one geometry is not an answer;
- physics sanity gates, pass/fail.

Metrics are computed on unstandardised values over reported timesteps only;
ensembles are averaged in physical space. Results are written to
`evaluation_part1.json` in the run directory, which is what the root-level
`select_best_model.py` and `collect_model_comparison.py` read.

### Where the numbers stand

- **Part1 (hash split).** Best available run `models_part1_r4_seed48`: combined
  valid+test TC R² **0.8526** (valid 0.8409 / test 0.8643, test RMSE 52.6 °C,
  HRR R² 0.941), chosen from 31 candidates with a **0.017** margin — inside the
  ±0.02 band, so best-available rather than significantly best.
- **The HRR head works** — R² 0.93–0.95 on total HRR across every run and split.
  `Q_TOTAL` is the exception (R² ≈ 0.12), but it is an energy-closure *residual*
  of ~18 kW RMSE against a ~4000 kW fire: numerically near noise, and labelled as
  such rather than reported as a failure.
- **`nocb` is the hardest geometry** in every architecture tried — with no cavity
  barriers the cavity flow is unobstructed and the plume path more chaotic, so
  point thermocouples are less predictable. Per-geometry n is small; directional.
- **60-sim v9 champion** is a 3-replicate population: valid R² 0.8297 ± 0.0040,
  test 0.8472 ± 0.0257, combined 0.8384 ± 0.0121 (`bestofn_v9_summary.txt`).
  Validation R² is stable; **test R² is the variance driver** — the 8-sim test
  bucket swings ~0.05 between identical retrains. Excluding the LES-chaotic
  `Test_1_PE_PIR` HRR1333 family, both valid and test clear 0.87 in all three
  replicates — a robust result, not a lucky draw.
- **KAN vs MLP is not established.** Compare 3-replicate populations, never single
  runs, and report any delta inside **±0.02 R² as inconclusive**.

---

## Streamlit app

```powershell
cd D:\VS_projects\bs8414_KAN_surrogate
.\run_app.ps1                 # http://localhost:8501
.\run_app.ps1 -Port 8502      # alongside the MLP app for a side-by-side
```

`run_app.ps1` activates this venv, picks `app_part1.py`, exports the material
table if it is missing, prints GPU status, then starts Streamlit. Manual
equivalent: `.\venv\Scripts\activate ; streamlit run app_part1.py`.

Pick a **cladding × insulation × geometry** build-up and it predicts the 16
external thermocouples and the 5-channel HRR budget over the 0–1800 s / 10 s
grid — auto-predicting on every change, under a second. Tabs: per-group TC curves
with an optional ±1 sd ensemble band and a peak table; the HRR budget with its
closure residual; a BR 135 external screen; and a data tab with CSV export and
the exact 16-d input vector. The material-property editor is prefilled with this
build-up's exact FDS values and can be edited to probe sensitivity.

The run selector ranks model directories by the recorded `combined_tc_r2` in
`evaluation_part1.json`; **★ selected** is `models_part1_r4_seed48`, the run whose
weights are kept out of `.gitignore` so a fresh clone can predict without
retraining. Currently 37 runs offered, 0 hidden.

**Prediction only** — the app never reads `D:\Bs8414_05052026`. Runtime inputs
are the checkpoint plus `app_assets/part1_materials.json`, written once by the
root-level `export_app_assets.py`, which refuses to write unless a
cladding/insulation id provably fixes its material block.

**Part1 enforced from the checkpoint, not the filename**: a run is offered only
if it carries 16 `sensor_names` *and* an HRR head. The 60-sim pipeline has 24
thermocouples and no HRR head, so it cannot pass — which is what keeps a 60-sim
checkpoint from loading into a 16-channel model and silently mislabelling
channels. Directories containing `BROKEN` are excluded outright. Anything hidden
is listed in the sidebar with the reason.

What the app refuses to claim:

- **Geometry cannot extrapolate** — an 8-way embedding over *observed* flag
  combinations. A build-up the corpus never contained gets a warning banner, not
  a quietly plausible curve. (`ACM_PE`+`PIR`, for instance, exists only as
  geometries 1, 2, 4, 5, 7 — the baseline is *not* in the corpus.)
- **BR 135 external only** — Part1 instruments the external face, so the internal
  fire-spread criterion cannot be assessed. The screen is a surrogate reading,
  not a classification and not a test result.
- **The ensemble band is member disagreement**, not a calibrated interval.
- **`Q_TOTAL` is a residual budget term** near numerical noise, labelled as such.

`app_common_part1.py`, `app_part1.py` and `run_app.ps1` are **byte-identical**
across the projects that hold them. Never hand-edit one copy; edit and re-copy.
Full app contract: `..\APPS.md`.

---

## Layout

```
config.py / config_part1.py       hyperparameters and paths — the source of truth
data_loader.py                    60-sim data layer
data_loader_part1.py              Part1 CHID/material/split logic       (shared file)
anchor_features.py                ignition-anchor feature bank
features_v4.py / features_v6.py   extended-FDS and physics-causal feature sets
model.py                          KANLinear + KANAttentionLSTM (v1; KANLinear is the origin block)
model_v3..v7.py                   the 60-sim version lineage
model_part1.py                    THE VARIABLE UNDER TEST on Part1
physics_part1.py                  physics gates + optional closure/geometry penalties (shared)
train.py, train_v2..v9.py         60-sim trainers, one per version
train_baseline63.py               original-baseline reference retrain
train_part1.py                    Part1 trainer                          (shared file)
evaluate_kan.py, evaluate_v5..v9.py   60-sim evaluators
evaluate_part1.py                 frozen Part1 eval contract             (shared file)
validate_physics_v6..v9.py        60-sim physics sanity gates
killer_excluded_v9.py             pooled R² with the LES-chaotic killer family excluded
br135_classification_v9.py        BR 135 external classification vs FDS truth
bestofn_v9_driver.py              best-of-N replicate driver
ablate_lambdareg_part1.py         spline-regulariser ablation
compare_part1.py / compare_splits.py / metrics_full_part1.py / time_inference_part1.py
dump_per_sensor*.py / dump_ts_part1.py    per-sensor and per-case dumps
explain_part1.py / causal_part1.py        SHAP and causal explainability (shared files)
verify_parity_part1.py            SHA-256 + array-hash proof of shared-layer identity
app_common_part1.py / app_part1.py / run_app.ps1    the Streamlit app
app_assets/                       part1_materials.json + selected_model.json
PART1_RESULTS.md / VARIANCE_RECORD.md / analysis_v*.md    the experimental record
```

### Repository layout

Run logs, analysis records and before-state snapshots are grouped so the project
root holds only what you run:

```
<project>/
  README.md            this file
  *.py                 all modules and entry points — flat, at the root
  models_*/            checkpoints + per-run provenance JSON
  app_assets/          part1_materials.json, selected_model.json
  docs/                results records and analyses (PART1_RESULTS.md, analysis_*.md, ...)
  logs/                paired .log / .err.log run logs — the provenance of every number
  archive/             before-state snapshots of deliberate edits (*.pre-*, *.bak)
```

**Python stays at the project root, deliberately.** Every module imports flat
(`from config_part1 import ...`) and `config.py` / `config_part1.py` derive
`PROJECT_DIR`, `MODEL_DIR`, `OUTPUT_DIR` and `SLICE_DIR` from `__file__` — moving
them into a `src/` package would silently repoint model and slice paths, and
those files must stay byte-identical across all eleven surrogate projects for
`verify_parity_part1.py` to pass. New run logs still land at the root; move them
into `logs/` when you tidy.

`CLAUDE.md` is git-ignored: it is the working brief for agent sessions, not part
of the published artefact.

---

## Judgment rules

- **Config is the source of truth** — read `config.py` / `config_part1.py` before
  changing a hyperparameter, and establish which corpus you are on first.
- **Never hand-edit a shared file.** `config_part1.py`, `data_loader_part1.py`,
  `train_part1.py`, `evaluate_part1.py`, `physics_part1.py`, `explain_part1.py`,
  `causal_part1.py` and the app files are byte-identical across projects; edit one
  copy and re-copy, then re-run `verify_parity_part1.py`.
- **Populations, not single runs.** Retrain variance on this corpus is real
  (±0.02–0.05 R² depending on split and bucket size). Report a delta inside
  ±0.02 as inconclusive, never as a win.
- **Physics gates first.** A better R² with a failing physics gate is reported as
  broken, not as an improvement.
- **State the split.** `hash` and `system` answer different questions.

## Related

`..\CLAUDE.md` (project map, Part1 contract) · `..\APPS.md` (app and deployment
contract) · `bs8414_MLP_surrogate` (the MLP ablation control — the reason this
project's blocks are the variable under test) · `bs8414_surrogate_model`
(Attention-LSTM V3 baseline) · `bs8414_fundiff_kan_surrogate`,
`bs8414_physicsnemo_kan_surrogate` (vendor `KANLinear` from here).
