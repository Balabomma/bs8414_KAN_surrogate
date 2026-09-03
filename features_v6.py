"""v6 physics-causal input features, appended to the v4 vector.

  dstar_dx    D*/dx — FDS resolution index. D* = (Q/(rho cp T sqrt(g)))^(2/5).
              The causal variable behind mesh-sensitivity bifurcation.
  t_ig_phys   Thermally-thick ignition delay estimate
              ~ k*rho*c (T_ref - T_amb)^2 / q''^2 (proportional form).
              Physics-based causal prior for ignition timing.
  anchor_conf Per-cladding anchor trustworthiness from train-side LOO
              residuals of the anchor regression: exp(-mean|resid|/0.05).

build_params_v6(params, meta, bank) -> (N, 16 + N_V4 + 3) input matrix.
"""
import numpy as np

from features_v4 import (
    build_params_v4, N_V4_FEATURES, V4_FEATURES, HRR_MW_SCALE,
)
from anchor_features import predict_anchor

N_V6_EXTRA = 3
N_V6_FEATURES = N_V4_FEATURES + N_V6_EXTRA

_HRR_MW_COL = 16 + V4_FEATURES.index("hrr_mw")

# ambient properties for D*
_RHO, _CP, _TAMB_K, _G = 1.204, 1.005, 291.0, 9.81
_DSTAR_DENOM = _RHO * _CP * _TAMB_K * np.sqrt(_G)

MESH_DX = {"M008": 0.08, "M009": 0.09, "M010": 0.10}


def _dstar_over_dx(hrr_mw, mesh_key):
    q_kw = hrr_mw * 1000.0
    dstar = (q_kw / _DSTAR_DENOM) ** 0.4
    return dstar / MESH_DX.get(mesh_key, 0.1)


def _t_ig_phys(core_k, core_rho, core_cp, core_tref, hrrpua_kw, reactive):
    """Proportional thermally-thick ignition delay; 1.0 = never ignites."""
    if not reactive or core_tref <= 20.0 or hrrpua_kw <= 0:
        return 1.0
    krc = core_k * 1e-3 * core_rho * core_cp   # kW-based k*rho*c
    t_ig = krc * (core_tref - 18.0) ** 2 / (hrrpua_kw ** 2)
    return float(np.clip(t_ig / 0.5, 0.0, 1.0))


def anchor_confidence_by_cladding(bank):
    """exp(-LOO mean|resid|/0.05) per cladding id, from the train bank."""
    conf = {}
    for c in range(4):
        pos = np.where(bank["clad"] == c)[0]
        if len(pos) < 3:
            conf[c] = 0.5
            continue
        resid = []
        for p in pos:
            pred = predict_anchor(c, float(bank["hrr"][p]), float(bank["mesh"][p]),
                                  bank, exclude_idx=int(p))
            resid.append(abs(pred - bank["anchor"][p]))
        conf[c] = float(np.exp(-np.mean(resid) / 0.05))
    return conf


def build_params_v6(params, meta, bank):
    """(N, 16) baseline -> (N, 16 + N_V4_FEATURES + 3) v6 input matrix."""
    p4 = build_params_v4(params, meta)
    conf = anchor_confidence_by_cladding(bank)

    extra = np.zeros((len(meta), N_V6_EXTRA), dtype=np.float32)
    for i, m in enumerate(meta):
        hrr_mw = float(p4[i, _HRR_MW_COL]) * HRR_MW_SCALE
        mat = m.get("material_features", [0.0] * 13)
        # material_features are normalised; recover raw core props
        core_cp = mat[0] * 4.0
        core_k = mat[1] * 0.5
        core_rho = mat[2] * 2000.0
        core_tref = mat[4] * 600.0
        reactive = mat[5] > 0.5
        hrrpua_kw = float(params[i, 1]) * 2333.3

        extra[i, 0] = _dstar_over_dx(hrr_mw, m.get("mesh", "M010")) / 25.0
        extra[i, 1] = _t_ig_phys(core_k, core_rho, core_cp, core_tref,
                                 hrrpua_kw, reactive)
        extra[i, 2] = conf.get(int(params[i, 0]), 0.5)

    return np.concatenate([p4, extra], axis=1)
