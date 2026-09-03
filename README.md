# bs8414_KAN_surrogate — KAN-Attention-LSTM thermocouple surrogate

A deep-learning surrogate for the **BS 8414-1** large-scale facade fire test.
Given a construction build-up, it predicts the external thermocouple response and
the global energy budget for the full 30-minute test in about a second, where the
FDS simulation it replaces takes 12–27 hours.

**Target:** 16 external thermocouples plus the 5-channel `_hrr.csv` energy budget
(`HRR, Q_RADI, Q_CONV, Q_COND, Q_TOTAL`) on the 0–1800 s / 10 s grid — 181
timesteps.

**Input:** a 16-d vector — `[cladding_id, insulation_id, geom_id]` plus 13
material properties parsed from the FDS deck.

---

## The corpus

The **Part1 geometry-variant corpus**, `D:\Bs8414_05052026\Part1\_completed`. It
is a geometry sensitivity study, so the design space is:

| Axis | Levels |
|---|---|
| Cladding | 12 systems (8 generic + 4 DCLG references) |
| Insulation | 5 products (MW, MWBC, PF, PIR, WC) |
| **Geometry** | **8** — the observed combinations of three construction modifiers |

HRR and mesh size are **constants** here (HRRPUA = 2333.3 kW/m², dx = 0.10 m,
16 meshes, T_END = 1800 s), so they carry no signal and are not inputs.

- 186 completed simulations → **184 usable**. Two are excluded by name in
  `config_part1.EXCLUDED_CHIDS`, each with its reason recorded rather than
  silently filtered by a shape check.
- **Hash split: 141 train / 20 valid / 23 test** (`PART1_SPLIT=hash`, default).
  Every base system appears in training, so this measures whether geometry
  effects are learned.
- `PART1_SPLIT=system` keeps all 8 geometry variants of a build-up together, so
  test systems are genuinely unseen. It answers a different question, and the two
  are **not comparable** — always state which one a number came from.
- Scalers are fitted on the **training rows only**.

### Why 16 thermocouples

169 of the 186 decks instrument the external face only; the `Insulation_LV2`
group survives in 17 legacy `DCLG_*` decks alone. Training an insulation head on
9 % of the corpus would be a fiction, so the target is the 16 external channels
in two grouped decoders (External LV1, External LV2). This costs the BR 135
**internal** fire-spread criterion entirely; the **external** criterion is
unaffected, and the app labels it as such.

### The HRR head

The burner ramp is identical in every deck, so the run-to-run variation in HRR
*is* the cladding/insulation combustion contribution — the same physics that
drives the thermocouples. That is why a second decoder predicts it from the same
temporal features instead of from a separate model. Weighted by
`LAMBDA_HRR = 0.3` (env-overridable).

`Q_TOTAL` is an energy-closure **residual** of ~18 kW RMSE against a ~4000 kW
fire. It sits at numerical noise and is reported as a budget term, not as a
physical prediction.

---

## The model

`model_part1.py` — **KAN-Attention-LSTM**,
`MODEL_NAME = "KAN-Attention-LSTM (Part1)"`, **850,765 parameters per member**,
`LAMBDA_REG = 2e-3`.

The distinguishing idea is **KAN**: every fully-connected layer in the parameter
encoder and the sensor decoders is a `KANLinear` — a learnable **B-spline
activation on each edge** (RBF basis, 8 knots, tanh-squashed inputs, SiLU
residual, per-layer LayerNorm) — rather than a fixed activation on the nodes. The
temporal backbone is conventional.

```
[cladding(12) + insulation(5) + geometry(8) embeddings + 13 material features]
     -> KAN parameter encoder
     -> sinusoidal TimeEncoding -> MultiScaleConv (k = 3 / 9 / 27)
     -> 2-layer BiLSTM(96) -> 4-head self-attention + LayerNorm
     -> KAN decoders:  External LV1 (8)   External LV2 (8)   HRR (5)
        with parameter -> output skip connections
```

**Geometry is a single 8-way embedding** over the observed flag combinations
(bit 0 `noair` — no ventilated cavity; bit 1 `nogap` — closed panel joints;
bit 2 `nocb` — no cavity barriers; id 0 = the fully-featured baseline). One
embedding rather than three booleans, so the model can learn an arbitrary
interaction between the modifiers instead of assuming they compose additively.
The cost, which constrains what may be asked of it afterwards: **a combination
absent from training has no embedding row and cannot be predicted at all.**

Physics is enforced **on the output**, not penalised in the loss — soft penalties
did not hold in earlier work. Thermocouples are clamped at ambient (18 °C) and
total HRR at zero. The spline regulariser (`kan_regularization`) adds spline L2
plus adjacent-knot smoothness across every `KANLinear`.

```powershell
python model_part1.py     # parameter count + forward-pass check
```

`layers_part1.py` holds the layer primitives — `KANLinear`, `TimeEncoding`,
`MultiScaleConv`, `KANSensorDecoder`, `kan_regularization` — vendored verbatim so
the pipeline stands alone. Verify a block against its origin with:

```powershell
python -c "import inspect, hashlib, layers_part1 as L; print(hashlib.sha256(inspect.getsource(L.KANLinear).encode()).hexdigest()[:16])"
```

---

## Training

Always from this directory, in **this project's own venv**, on the NVIDIA GPU.

```powershell
cd D:\VS_projects\bs8414_KAN_surrogate
.\venv\Scripts\activate
nvidia-smi                                  # confirm the 4090 is free
python verify_parity_part1.py               # data layer byte-identical
python -u train_part1.py --members 3 --seed 48 --model-dir models_part1_r5 `
       > logs\train_part1_r5.log 2> logs\train_part1_r5.err.log
python evaluate_part1.py --model-dir models_part1_r5
```

| Flag | Default | Meaning |
|---|---|---|
| `--model-dir` | `models_part1` | output directory — **must not already hold a run** |
| `--members` | 1 | ensemble members; the standard protocol is **3**, seeded `seed + m_idx` |
| `--seed` | 42 | base seed; the balanced design uses 42 / 45 / 48 / 52 / 55 / 61 |
| `--split` | `hash` | `hash` or `system` — different experiments, not comparable |
| `--epochs` | 500 | max epochs; early stopping, patience 60 |
| `--force` | off | overwrite a directory that already holds checkpoints |

Roughly **2.5 minutes per member** on a 4090, so a 3-member ensemble is ~8
minutes.

**Two safety features exist because of real defects.** The trainer refuses to
write into a directory that already holds checkpoints — three retrains were lost
to silent overwrites — so `--force` is deliberate, never routine. And every log
carries a `[sentinel] |dW| = ...` line after epoch 0 proving the weights actually
moved, because a refactor once pulled `zero_grad/backward/step` inside a
regulariser guard and a model trained for zero steps while reporting "best @
epoch 0", which reads exactly like instant convergence.

Environment knobs: `PART1_SPLIT`, `PART1_LAMBDA_HRR` (0.3),
`PART1_LAMBDA_CLOSURE` and `PART1_LAMBDA_GEOM` (0.0 — the physics ablations),
`PART1_SIMS_DIR`.

`ablate_lambdareg_part1.py` retrains with the spline regulariser switched off,
for the question of whether it does any work.

### Reading the run directories

Naming is systematic, and the log beside each run is its provenance.

| Pattern | What it is |
|---|---|
| `models_part1_r1` … `r4_seed48`, `r4_s52` | main sequence |
| `models_part1_kanbal_s{42,45,48,52,55,61}_r{1,2,3}` | the **balanced design** — 3 independent retrains at each of several base seeds |
| `models_part1_kan_noreg_seed*`, `models_part1_noreg_seed42` | spline-regulariser ablation |
| `models_part1_closure_w01_s48`, `_w05_s48` | energy-closure penalty ablation |
| `models_part1_sys_kan_seed*` | `PART1_SPLIT=system` — **not comparable** to hash runs |
| `models_part1_184_r1`, `*_c184`, `models_part1_final_r1` | runs on the corrected 184-configuration corpus |

The balanced design exists because one spread number conflates two things:
**within-seed** variance (cuDNN non-determinism leaves the per-member draw
random, ~0.018) and **between-seed** variance (~0.062, about 3.4× larger). An
unbalanced, seed-42-heavy pool has a composition-dependent mean, so runs are
reported by seed cell.

---

## Evaluation

```powershell
python evaluate_part1.py     --model-dir models_part1_r4_seed48
python metrics_full_part1.py --model-dir models_part1_r4_seed48 --split test
python compare_part1.py                      # replicate ensembles -> population
python time_inference_part1.py               # inference cost on the test split
python dump_ts_part1.py                      # per-case time series
python explain_part1.py                      # SHAP attribution
python causal_part1.py                       # interventional explainability
```

`evaluate_part1.py` is a **frozen contract**, fixed before any candidate model
existed. A candidate needing different scoring is a different experiment. Per
split it reports:

- pooled and per-group R² / RMSE on the 16 thermocouples, in °C;
- R² / RMSE per HRR channel, in kW;
- a **per-geometry breakdown** — the point of this corpus is whether removing the
  cavity, the gaps or the barriers is predictable, so a pooled number that hides
  a failure on one geometry is not an answer;
- physics sanity gates, pass/fail.

Metrics are computed on unstandardised values over reported timesteps only, and
ensembles are averaged in physical space. Results land in
`evaluation_part1.json` inside the run directory.

### Where the numbers stand

- **Best available run: `models_part1_r4_seed48`** — combined valid+test TC R²
  **0.8526** (valid 0.8409 / test 0.8643, test RMSE 52.6 °C, HRR R² 0.941),
  selected from 31 candidates on the hash split. The margin over the runner-up is
  **0.017**, inside the ±0.02 band this work treats as inconclusive — best
  available, not significantly best.
- **The HRR head works** — R² 0.93–0.95 on total HRR across every run and split.
- **`nocb` is the hardest geometry.** With no cavity barriers the cavity flow is
  unobstructed and the plume path more chaotic, so point thermocouples are less
  predictable. Per-geometry n is small on test; directional.

Physics gates must pass before any accuracy claim counts. The growth-monotonicity
gate compares the prediction against the ground truth on the same cases and asks
whether the model is *less* physical than the simulation it imitates — an earlier
absolute version failed on the FDS data itself, because an LES point thermocouple
is not monotonic at 10 s sampling.

---

## Streamlit app

```powershell
cd D:\VS_projects\bs8414_KAN_surrogate
.\run_app.ps1                 # http://localhost:8501
.\run_app.ps1 -Port 8502      # a second instance alongside the first
```

`run_app.ps1` activates the venv, exports the material table if missing, prints
GPU status, then starts Streamlit. Manual equivalent:
`.\venv\Scripts\activate ; streamlit run app_part1.py`.

Pick a **cladding × insulation × geometry** build-up and it predicts the 16
thermocouples and the HRR budget, auto-updating on every change in under a
second. Tabs: per-group TC curves with an optional ±1 sd ensemble band and a peak
table; the HRR budget with its closure residual; a BR 135 external screen; and a
data tab with CSV export and the exact 16-d input vector. The material-property
editor is prefilled with the build-up's exact FDS values and can be edited to
probe sensitivity.

The run picker ranks directories by the recorded `combined_tc_r2`;
**★ selected** is `models_part1_r4_seed48`, whose weights are the only ones kept
out of `.gitignore` so a fresh clone can predict without retraining.

**Prediction only** — the app never reads `D:\Bs8414_05052026`. Its runtime
inputs are the checkpoint and `app_assets/part1_materials.json`, written once by
an export step that refuses to write unless a cladding/insulation id provably
fixes its material block, so the shipped table is exact rather than an average
over two systems sharing a name.

**Part1 is enforced from the checkpoint, not the filename**: a run is offered
only if it carries 16 `sensor_names` *and* an HRR head. Directories containing
`BROKEN` are excluded outright. Anything hidden is listed in the sidebar with the
reason, so a missing run is visible rather than mysterious.

What the app refuses to claim:

- **Geometry cannot extrapolate** — an 8-way embedding over *observed* flag
  combinations. A build-up the corpus never contained gets a warning banner, not
  a quietly plausible curve.
- **BR 135 external only** — a surrogate reading, not a classification and not a
  test result.
- **The ensemble band is member disagreement**, not a calibrated interval.
- **`Q_TOTAL` is a residual budget term** near numerical noise, labelled as such.

---

## Layout

```
config_part1.py             corpus contract: design space, split, targets, exclusions
data_loader_part1.py        CHID parsing, material extraction, split assignment
layers_part1.py             KANLinear, TimeEncoding, MultiScaleConv, KANSensorDecoder
model_part1.py              the architecture
train_part1.py              trainer
evaluate_part1.py           frozen evaluation contract
physics_part1.py            physics gates + optional closure/geometry penalties
ablate_lambdareg_part1.py   spline-regulariser ablation
compare_part1.py            replicate ensembles -> population comparison
metrics_full_part1.py       additional descriptive metrics
time_inference_part1.py     inference cost
dump_ts_part1.py            per-case time series for the test split
dump_per_sensor_ext.py      per-sensor test metrics
explain_part1.py            SHAP attribution
causal_part1.py             interventional / causal explainability
verify_parity_part1.py      SHA-256 + array-hash proof of the shared data layer
app_common_part1.py         shared Streamlit input layer
app_part1.py                the Streamlit app
run_app.ps1                 app launcher
app_assets/                 part1_materials.json + selected_model.json
models_part1_*/             checkpoints + per-run provenance JSON
docs/                       PART1_RESULTS.md — the results record
logs/                       paired .log / .err.log — provenance of every number
```

Python sits at the project root deliberately: modules import flat
(`from config_part1 import ...`), and `config_part1.py` derives `PROJECT_DIR`,
`MODEL_DIR` and `SLICE_DIR` from `__file__`, so a `src/` package would silently
repoint model and data paths.

Weights are git-ignored by extension rather than by directory, so per-run
`evaluation_part1.json` and `history_member*.json` stay tracked — git cannot
re-include a file inside an excluded directory. Run logs are tracked on purpose:
they are the provenance of every number in `docs/PART1_RESULTS.md`.

---

## Working rules

- **`config_part1.py` is the source of truth** for the corpus; read it before
  changing anything.
- **Never hand-edit a shared file.** The data layer is byte-identical across the
  surrogate family; edit one copy, re-copy, then re-run `verify_parity_part1.py`.
- **Populations, not single runs.** Report a delta inside **±0.02 R²** as
  inconclusive, never as a win.
- **Physics gates before accuracy claims.** A better R² with a failing gate is
  reported as broken.
- **State the split.** `hash` and `system` answer different questions.
