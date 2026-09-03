r"""BS 8414 Part1 thermocouple surrogate — Streamlit prediction app.

    venv\Scripts\activate  &&  streamlit run app_part1.py

Predicts the 16 external thermocouples and the 5-channel `_hrr.csv` energy
budget over the 0-1800 s / 10 s grid for any cladding x insulation x geometry
build-up, straight from a trained Part1 ensemble.

SHARED FILE — byte-identical in `bs8414_KAN_surrogate` and
`bs8414_MLP_surrogate`. It binds only to the uniform Part1 interface
(`Part1Surrogate`, `MODEL_NAME` from `model_part1.py`), which is the one file
that differs between them, so the same app serves either architecture and any
difference you see is attributable to the architecture. Never hand-edit one
copy: edit this file and re-copy it.

Runtime dependencies are the checkpoint plus `app_assets/part1_materials.json`
(written by `..\export_app_assets.py`). The FDS corpus is NOT read — the app
runs on a machine that has never seen `D:\Bs8414_05052026`.

What the app deliberately does not claim:
  * Geometry is an 8-way embedding over observed flag combinations. A build-up
    the corpus never contained is an extrapolation, and the app says so rather
    than quietly returning a curve that looks plausible.
  * Part1 instruments the external face only, so the BR 135 *external*
    fire-spread screen is computable and the *internal* one is not. The screen
    is a surrogate reading, not a test result.
  * The ensemble band is member disagreement, not a calibrated interval.
"""
import glob
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch

# Launch-context guard: `streamlit run` puts this directory on sys.path, but
# other entry points (the headless AppTest harness, `python -m`) do not. Same
# idiom as `model_part1.py` / `dataset_part1.py`.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app_common_part1 import (
    TIME_S, apply_plot_style, asset_caption, build_params, build_up_selector,
    input_vector_table, load_selection, material_editor, require_assets,
    selection_caption, selection_index, selection_label,
)
from config_part1 import (
    DEVICE, DT_DEVC, GEOMETRY_NAMES, GROUP_SIZES, HRR_CHANNELS, N_SENSORS,
    N_TIMESTEPS, SENSOR_GROUPS, T_AMBIENT, T_END,
)
from data_loader_part1 import ChannelScaler
from model_part1 import MODEL_NAME, Part1Surrogate

apply_plot_style()

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

# BR 135 external fire-spread screen, same thresholds as `_br135.py`:
# start when mean External LV1 first exceeds ambient + 200 degC; fail if any
# External LV2 channel holds ambient + 600 degC for 30 s within 900 s of start.
BR135_START_RISE = 200.0
BR135_FAIL_RISE = 600.0
BR135_SUSTAIN_STEPS = 3
BR135_WINDOW_STEPS = int(900.0 / DT_DEVC)


# ── checkpoints ───────────────────────────────────────────────────────────

def _part1_signature(ckpt):
    """Why this checkpoint is (or is not) a Part1 model, as (ok, reason).

    The app serves the **Part1 geometry corpus only**. Relying on the
    `*member*.pt` filename — which happens to be unique to the Part1 trainer —
    would be a convention, not a check, and a 60-sim checkpoint that ever
    acquired that name would load into a 16-channel model and silently mislabel
    24 thermocouples. So the corpus is verified from the checkpoint's own
    contents: Part1 carries the 16 external channels and the HRR head, the
    60-sim pipeline has 24 channels and no HRR head at all.
    """
    names = ckpt.get("sensor_names")
    if not isinstance(names, (list, tuple)):
        return False, "no sensor_names — not written by the Part1 trainer"
    if len(names) != N_SENSORS:
        return False, (f"{len(names)} thermocouples, not {N_SENSORS} — "
                       f"60-sim corpus")
    if "hrr_scaler" not in ckpt or "tc_scaler" not in ckpt:
        return False, "no HRR head — 60-sim corpus"
    return True, ""


@st.cache_data(show_spinner=False)
def discover_runs():
    """Part1 ensembles in this project, best recorded score first.

    Returns (runs, rejected). Anything that is not a Part1 model, and anything
    the project has marked BROKEN, is kept out of the picker and reported.
    """
    runs, rejected = [], []
    for path in sorted(glob.glob(os.path.join(PROJECT_DIR, "models*"))):
        members = sorted(glob.glob(os.path.join(path, "*member*.pt")))
        if not members:
            continue
        name = os.path.basename(path)

        # Runs the project itself has labelled defective must never be one
        # click away from a prediction.
        if "BROKEN" in name.upper():
            rejected.append((name, "marked BROKEN by the project"))
            continue

        ckpt = torch.load(members[0], map_location="cpu", weights_only=False)
        ok, why = _part1_signature(ckpt)
        if not ok:
            rejected.append((name, why))
            continue

        score = None
        eval_json = os.path.join(path, "evaluation_part1.json")
        if os.path.isfile(eval_json):
            try:
                with open(eval_json) as f:
                    score = json.load(f).get("combined_tc_r2")
            except (ValueError, OSError):
                score = None
        runs.append({"name": name, "path": path, "n_members": len(members),
                     "combined_tc_r2": score, "split_mode": ckpt.get("split_mode"),
                     "mtime": os.path.getmtime(members[-1])})
    runs.sort(key=lambda r: (r["combined_tc_r2"] is None,
                             -(r["combined_tc_r2"] or 0.0), -r["mtime"]))
    return runs, rejected


@st.cache_resource(show_spinner=False)
def load_ensemble(model_dir):
    """Every member of one run, plus the scalers frozen into its checkpoints."""
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    models, meta = [], None
    for path in sorted(glob.glob(os.path.join(model_dir, "*member*.pt"))):
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model = Part1Surrogate().to(device)
        # The ambient and HRR floors ride in the state_dict as buffers, so the
        # physical clamps are restored without re-calling set_output_scaling.
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        models.append(model)
        meta = ckpt
    tc_scaler = ChannelScaler().load_state_dict(meta["tc_scaler"])
    hrr_scaler = ChannelScaler().load_state_dict(meta["hrr_scaler"])
    return models, tc_scaler, hrr_scaler, meta, device


@torch.no_grad()
def predict(models, params_vec, tc_scaler, hrr_scaler, device):
    """(members, 181, 16) degC and (members, 181, 5) kW — physical units."""
    params = torch.as_tensor(params_vec, dtype=torch.float32,
                             device=device).unsqueeze(0)
    time_array = torch.as_tensor(np.linspace(0.0, 1.0, N_TIMESTEPS),
                                 dtype=torch.float32, device=device)
    tc_members, hrr_members = [], []
    for model in models:
        tc, hrr, _ = model(params, time_array)
        tc_members.append(tc_scaler.inverse(tc.cpu().numpy())[0])
        hrr_members.append(hrr_scaler.inverse(hrr.cpu().numpy())[0])
    return np.stack(tc_members), np.stack(hrr_members)


# ── readouts ──────────────────────────────────────────────────────────────

def br135_external(tc):
    """(fail, t_start_s, t_fail_s, worst_channel) from one (181, 16) array."""
    lv1 = tc[:, :GROUP_SIZES[0]].mean(axis=1)
    lv2 = tc[:, GROUP_SIZES[0]:]
    start = np.where(lv1 >= T_AMBIENT + BR135_START_RISE)[0]
    if len(start) == 0:
        return False, None, None, None
    ts = int(start[0])
    hot = lv2 >= T_AMBIENT + BR135_FAIL_RISE
    stop = min(ts + BR135_WINDOW_STEPS + 1, lv2.shape[0])
    for j in range(lv2.shape[1]):
        run = 0
        for t in range(ts, stop):
            run = run + 1 if hot[t, j] else 0
            if run >= BR135_SUSTAIN_STEPS:
                return True, TIME_S[ts], TIME_S[t], j
    return False, TIME_S[ts], None, None


def curve_figure(mean, spread, names, title, show_band):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    colors = plt.cm.turbo(np.linspace(0.05, 0.95, mean.shape[1]))
    for j, name in enumerate(names):
        ax.plot(TIME_S, mean[:, j], lw=1.4, color=colors[j],
                label=name.split("(")[0])
        if show_band and spread is not None:
            ax.fill_between(TIME_S, mean[:, j] - spread[:, j],
                            mean[:, j] + spread[:, j],
                            color=colors[j], alpha=0.15, lw=0)
    ax.axhline(T_AMBIENT, color="0.5", ls=":", lw=1.0)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Temperature (\u00b0C)")
    ax.set_title(title)
    ax.set_xlim(0, T_END)
    ax.legend(ncol=2, loc="upper left")
    fig.tight_layout()
    return fig


def hrr_figure(mean, spread, show_band):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    axes[0].plot(TIME_S, mean[:, 0], lw=1.8, color="#c0392b", label="HRR")
    if show_band and spread is not None:
        axes[0].fill_between(TIME_S, mean[:, 0] - spread[:, 0],
                             mean[:, 0] + spread[:, 0],
                             color="#c0392b", alpha=0.2, lw=0)
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Heat release rate (kW)")
    axes[0].set_title("Total HRR")
    axes[0].set_xlim(0, T_END)
    axes[0].legend(loc="upper left")

    for j, channel in enumerate(HRR_CHANNELS[1:], start=1):
        axes[1].plot(TIME_S, mean[:, j], lw=1.3, label=channel)
    axes[1].axhline(0.0, color="0.5", ls=":", lw=1.0)
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Energy budget term (kW)")
    axes[1].set_title("Budget channels")
    axes[1].set_xlim(0, T_END)
    axes[1].legend(loc="lower left", ncol=2)
    fig.tight_layout()
    return fig


# ── app ───────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title=f"BS 8414 Part1 \u2014 {MODEL_NAME}",
                       page_icon="\U0001f525", layout="wide")
    st.title("BS 8414-1 facade fire \u2014 Part1 thermocouple surrogate")
    st.caption(f"{MODEL_NAME} \u2014 16 external thermocouples + 5-channel energy "
               f"budget, 0\u2013{int(T_END)} s at {int(DT_DEVC)} s")

    assets = require_assets()
    runs, rejected = discover_runs()
    if not runs:
        skipped = ("; ".join(f"`{n}` ({why})" for n, why in rejected)
                   or "none found")
        st.error(f"No Part1 ensemble found in `{PROJECT_DIR}` — a run must carry "
                 f"the {N_SENSORS} external thermocouples and the HRR head. "
                 f"Directories skipped: {skipped}.\n\nTrain one with "
                 f"`python train_part1.py --model-dir models_part1_r1`.")
        st.stop()

    # ── sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("Model")
        selection = load_selection()
        labels = {
            r["name"]: (f"{selection_label(r['name'], selection)}  \u2014 "
                        f"{r['n_members']} member"
                        f"{'s' if r['n_members'] != 1 else ''}"
                        + (f", R\u00b2 {r['combined_tc_r2']:.4f}"
                           if r["combined_tc_r2"] is not None else ""))
            for r in runs
        }
        names = [r["name"] for r in runs]
        choice = st.selectbox("Trained run", names,
                              index=selection_index(names, selection),
                              format_func=lambda n: labels[n])
        run = next(r for r in runs if r["name"] == choice)
        st.caption(selection_caption(selection, choice))

        with st.spinner("Loading ensemble\u2026"):
            models, tc_scaler, hrr_scaler, ckpt, device = load_ensemble(run["path"])

        st.caption(
            f"device **{device.type}**"
            + (f" \u00b7 {torch.cuda.get_device_name(0)}"
               if device.type == "cuda" else "")
            + f"  \n{len(models)} member(s) \u00b7 {ckpt['n_params']:,} parameters"
            f"  \nsplit mode **{ckpt['split_mode']}** \u00b7 seed {ckpt['seed']}"
        )
        if run["combined_tc_r2"] is not None:
            st.caption(f"recorded combined valid+test TC R\u00b2 "
                       f"**{run['combined_tc_r2']:.4f}** "
                       f"(`evaluation_part1.json`)")
        else:
            st.caption("no `evaluation_part1.json` in this run \u2014 score unknown; "
                       f"run `evaluate_part1.py --model-dir {run['name']}`")
        if rejected:
            with st.expander(f"{len(rejected)} directory/-ies hidden"):
                for n, why in rejected:
                    st.caption(f"`{n}` \u2014 {why}")

        st.divider()
        show_band = st.toggle("Ensemble spread (\u00b11 sd)", value=len(models) > 1,
                              disabled=len(models) < 2,
                              help="Spread across ensemble members. It measures "
                                   "member disagreement, not a calibrated "
                                   "uncertainty interval.")
        st.divider()
        st.caption(asset_caption(assets))

    # ── inputs ────────────────────────────────────────────────────────────
    cladding, insulation, geom_id, _ = build_up_selector(assets)
    material_raw = material_editor(assets, cladding, insulation)
    params_vec = build_params(assets, cladding, insulation, geom_id, material_raw)

    tc_members, hrr_members = predict(models, params_vec, tc_scaler,
                                      hrr_scaler, device)
    tc_mean, hrr_mean = tc_members.mean(axis=0), hrr_members.mean(axis=0)
    tc_sd = tc_members.std(axis=0) if len(models) > 1 else None
    hrr_sd = hrr_members.std(axis=0) if len(models) > 1 else None
    sensor_names = ckpt["sensor_names"]

    # ── headline ──────────────────────────────────────────────────────────
    st.subheader("Prediction")
    lv1 = tc_mean[:, :GROUP_SIZES[0]]
    lv2 = tc_mean[:, GROUP_SIZES[0]:]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Peak External LV1", f"{lv1.max():.0f} \u00b0C",
              help=f"at t = {TIME_S[lv1.max(axis=1).argmax()]:.0f} s")
    m2.metric("Peak External LV2", f"{lv2.max():.0f} \u00b0C",
              help=f"at t = {TIME_S[lv2.max(axis=1).argmax()]:.0f} s")
    m3.metric("Peak HRR", f"{hrr_mean[:, 0].max():.0f} kW",
              help=f"at t = {TIME_S[hrr_mean[:, 0].argmax()]:.0f} s")
    fail, t_start, t_fail, worst = br135_external(tc_mean)
    m4.metric("BR 135 external screen", "FAIL" if fail else "PASS",
              help="Surrogate reading of the external fire-spread criterion, "
                   "not a test result.")

    tabs = st.tabs(["Thermocouples", "Energy budget", "BR 135 screen", "Data"])

    with tabs[0]:
        start = 0
        for group, size in zip(SENSOR_GROUPS, GROUP_SIZES):
            sl = slice(start, start + size)
            fig = curve_figure(tc_mean[:, sl],
                               tc_sd[:, sl] if tc_sd is not None else None,
                               sensor_names[sl], group.replace("_", " "),
                               show_band)
            st.pyplot(fig)
            plt.close(fig)
            start += size
        st.dataframe(pd.DataFrame([{
            "thermocouple": name,
            "peak (\u00b0C)": round(float(tc_mean[:, j].max()), 1),
            "time to peak (s)": float(TIME_S[tc_mean[:, j].argmax()]),
            "final (\u00b0C)": round(float(tc_mean[-1, j]), 1),
            "member sd at peak (\u00b0C)": (
                round(float(tc_sd[tc_mean[:, j].argmax(), j]), 1)
                if tc_sd is not None else None),
        } for j, name in enumerate(sensor_names)]),
            width="stretch", hide_index=True)

    with tabs[1]:
        fig = hrr_figure(hrr_mean, hrr_sd, show_band)
        st.pyplot(fig)
        plt.close(fig)
        residual = hrr_mean[:, 4] - hrr_mean[:, :4].sum(axis=1)
        st.caption(
            f"Energy-closure residual `Q_TOTAL - (HRR + Q_RADI + Q_CONV + "
            f"Q_COND)`: max |residual| {np.abs(residual).max():.1f} kW against a "
            f"{hrr_mean[:, 0].max():.0f} kW peak fire. `Q_TOTAL` is a residual "
            f"budget term near numerical noise \u2014 it is the weakest channel in "
            f"every recorded run and should not be read as a physical quantity.")
        st.dataframe(pd.DataFrame({
            "channel": HRR_CHANNELS,
            "peak (kW)": [round(float(hrr_mean[:, j].max()), 1)
                          for j in range(len(HRR_CHANNELS))],
            "min (kW)": [round(float(hrr_mean[:, j].min()), 1)
                         for j in range(len(HRR_CHANNELS))],
            "final (kW)": [round(float(hrr_mean[-1, j]), 1)
                           for j in range(len(HRR_CHANNELS))],
        }), width="stretch", hide_index=True)

    with tabs[2]:
        st.markdown(
            f"**External fire-spread screen** \u2014 start when mean External LV1 "
            f"first exceeds ambient + {BR135_START_RISE:.0f} \u00b0C; fail if any "
            f"External LV2 channel holds ambient + {BR135_FAIL_RISE:.0f} \u00b0C "
            f"for {BR135_SUSTAIN_STEPS * DT_DEVC:.0f} s within "
            f"{BR135_WINDOW_STEPS * DT_DEVC:.0f} s of start.")
        if t_start is None:
            st.info("External LV1 never reaches the start threshold, so the "
                    "criterion never begins. No classification.")
        else:
            st.write(f"- start of level-2 exposure: **t = {t_start:.0f} s**")
            if fail:
                st.error(f"FAIL \u2014 `{sensor_names[GROUP_SIZES[0] + worst]}` "
                         f"sustained the threshold from **t = {t_fail:.0f} s**.")
            else:
                st.success(
                    f"PASS \u2014 no External LV2 channel sustained "
                    f"{T_AMBIENT + BR135_FAIL_RISE:.0f} \u00b0C; the hottest LV2 "
                    f"reading was {lv2.max():.0f} \u00b0C.")
        st.warning(
            "This is the surrogate's reading of the **external** criterion only. "
            "The Part1 corpus instruments the external face, so the BR 135 "
            "**internal** criterion cannot be assessed at all. A screen from a "
            "surrogate is not a classification and is not a test result.")

    with tabs[3]:
        tc_df = pd.DataFrame(tc_mean, columns=sensor_names)
        tc_df.insert(0, "time_s", TIME_S)
        hrr_df = pd.DataFrame(hrr_mean, columns=HRR_CHANNELS)
        hrr_df.insert(0, "time_s", TIME_S)
        stem = f"{cladding}_{insulation}_{GEOMETRY_NAMES[geom_id]}"
        c1, c2 = st.columns(2)
        c1.download_button("Download thermocouples (CSV)",
                           tc_df.to_csv(index=False).encode(),
                           file_name=f"{stem}_tc.csv", mime="text/csv")
        c2.download_button("Download energy budget (CSV)",
                           hrr_df.to_csv(index=False).encode(),
                           file_name=f"{stem}_hrr.csv", mime="text/csv")
        st.markdown("**Input vector (16-d, as fed to the model)**")
        st.dataframe(pd.DataFrame(input_vector_table(
            params_vec, cladding, insulation, geom_id, material_raw)),
            width="stretch", hide_index=True)
        st.dataframe(tc_df, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
