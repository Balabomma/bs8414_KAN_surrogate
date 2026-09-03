"""Extended FDS-derived input features for the v4 surrogate.

Adds to the baseline 16 inputs (cladding_id, hrr_norm, mesh_norm, 13 material
features) everything else in the FDS scripts that varies across simulations,
plus physics-derived quantities:

  Raw FDS (11): core emissivity, core heating rate, core reference rate,
      core CO yield, ins reference rate, ins pyrolysis range,
      ins-gas HoC, ins-gas CO yield, ins-gas soot yield,
      ins-gas specific heat, ins-gas enthalpy of formation
  Derived (8): core/ins thermal inertia sqrt(k rho cp), thermal diffusivity
      k/(rho cp), fire load HoR*rho/1e6; hrr_inv (1/hrr_norm, ignition-delay
      scale); hrr*mesh interaction

All features are normalised to ~[0, 1] with fixed ranges so the vector is
stable when new simulations are added.
"""
import re
import numpy as np

V4_FEATURES = [
    "core_emissivity", "core_heating_rate", "core_reference_rate", "core_co_yield",
    "ins_reference_rate", "ins_pyrolysis_range",
    "insgas_hoc", "insgas_co_yield", "insgas_soot_yield",
    "insgas_specific_heat", "insgas_enthalpy_formation",
    "core_thermal_inertia", "core_thermal_diffusivity", "core_fire_load",
    "ins_thermal_inertia", "ins_thermal_diffusivity", "ins_fire_load",
    "hrr_inv", "hrr_mesh", "hrr_mw",
]
N_V4_FEATURES = len(V4_FEATURES)

V4_NORM_RANGES = {
    "core_emissivity": (0, 1.0),
    "core_heating_rate": (0, 60.0),
    "core_reference_rate": (0, 0.01),
    "core_co_yield": (0, 0.05),
    "ins_reference_rate": (0, 0.002),
    "ins_pyrolysis_range": (0, 150.0),
    "insgas_hoc": (0, 30000.0),
    "insgas_co_yield": (0, 0.05),
    "insgas_soot_yield": (0, 0.2),
    "insgas_specific_heat": (0, 2.0),
    "insgas_enthalpy_formation": (-6.0e4, 0.0),
    "core_thermal_inertia": (0, 45.0),        # sqrt(kJ-units k*rho*cp)
    "core_thermal_diffusivity": (0, 2.0e-4),
    "core_fire_load": (0, 4.0),               # HoR*rho/1e6
    "ins_thermal_inertia": (0, 3.0),
    "ins_thermal_diffusivity": (0, 2.0e-3),
    "ins_fire_load": (0, 0.1),
    "hrr_inv": (0.5, 1.8),                    # 1/hrr_norm in [1, 1.75]
    "hrr_mesh": (0.4, 1.0),                   # hrr_norm*mesh_norm
    "hrr_mw": (0.0, 4.0),                     # HRRPUA * burner area / 1000 [MW]
}

HRR_MW_SCALE = 4.0     # denormalisation factor for the hrr_mw feature
HRRPUA_REF = 2333.3    # hrr_norm = HRRPUA / HRRPUA_REF
DEFAULT_FIRE_AREA = 1.5  # m^2, BS8414 burner crib (1.5 x 1.0 m OBST)


def _block(content, kind, id_pat):
    m = re.search(rf"&{kind}\s+ID\s*=\s*'{id_pat}'.*?/", content,
                  re.DOTALL | re.IGNORECASE)
    return m.group(0) if m else None


def _num(blk, key, default=0.0):
    if blk is None:
        return default
    m = re.search(rf"{key}\s*=\s*([\d.Ee+-]+)", blk)
    return float(m.group(1)) if m else default


def _matl(content, pats):
    for p in pats:
        b = _block(content, "MATL", p)
        if b:
            return b
    return None


def _reac_by_fuel(content, fuels):
    for f in fuels:
        m = re.search(rf"&REAC[^/]*?FUEL\s*=\s*'{f}'.*?/", content,
                      re.DOTALL | re.IGNORECASE)
        if m:
            return m.group(0)
    return None


def extract_v4_raw(fds_path, hrr_norm, mesh_norm):
    """Returns dict of raw (unnormalised) v4 feature values."""
    try:
        with open(fds_path, "r") as f:
            content = f.read()
    except Exception:
        content = ""

    # burner surface area from the Fire_source obstruction (x-y footprint)
    fire_area = DEFAULT_FIRE_AREA
    m = re.search(r"&OBST ID='Fire_source', XB=([\d.,\s-]+)", content)
    if m:
        xb = [float(x) for x in m.group(1).split(",")[:6]]
        if len(xb) >= 4:
            fire_area = (xb[1] - xb[0]) * (xb[3] - xb[2])
    fire_surf = _block(content, "SURF", "Fire")
    hrrpua = _num(fire_surf, "HRRPUA", hrr_norm * HRRPUA_REF)

    core = _matl(content, ["Core_MATERIAL", "Core_FRPE", "Core_PE"])
    ins = _matl(content, ["Insulation_VIRGIN", "Phenolic_VIRGIN", "PIR_VIRGIN",
                          "Insulation"])
    core_reac = _reac_by_fuel(content, ["ETHYLENE", "Phenolic_GAS"])
    ins_reac = _reac_by_fuel(content, ["Insulation_GAS"])
    spec = _block(content, "SPEC", "Insulation_GAS")

    core_cp = _num(core, "SPECIFIC_HEAT")
    core_k = _num(core, "CONDUCTIVITY")
    core_rho = _num(core, "DENSITY")
    core_hor = _num(core, "HEAT_OF_REACTION")
    ins_cp = _num(ins, "SPECIFIC_HEAT")
    ins_k = _num(ins, "CONDUCTIVITY")
    ins_rho = _num(ins, "DENSITY")
    ins_hor = _num(ins, "HEAT_OF_REACTION")

    def inertia(k, rho, cp):
        return float(np.sqrt(max(k * rho * cp, 0.0)))

    def diffus(k, rho, cp):
        d = rho * cp
        return k / d if d > 0 else 0.0

    return {
        "core_emissivity": _num(core, "EMISSIVITY"),
        "core_heating_rate": _num(core, "HEATING_RATE"),
        "core_reference_rate": _num(core, "REFERENCE_RATE"),
        "core_co_yield": _num(core_reac, "CO_YIELD"),
        "ins_reference_rate": _num(ins, "REFERENCE_RATE"),
        "ins_pyrolysis_range": _num(ins, "PYROLYSIS_RANGE"),
        "insgas_hoc": _num(ins_reac, "HEAT_OF_COMBUSTION"),
        "insgas_co_yield": _num(ins_reac, "CO_YIELD"),
        "insgas_soot_yield": _num(ins_reac, "SOOT_YIELD"),
        "insgas_specific_heat": _num(spec, "SPECIFIC_HEAT"),
        "insgas_enthalpy_formation": _num(spec, "ENTHALPY_OF_FORMATION"),
        "core_thermal_inertia": inertia(core_k, core_rho, core_cp),
        "core_thermal_diffusivity": diffus(core_k, core_rho, core_cp),
        "core_fire_load": core_hor * core_rho / 1e6,
        "ins_thermal_inertia": inertia(ins_k, ins_rho, ins_cp),
        "ins_thermal_diffusivity": diffus(ins_k, ins_rho, ins_cp),
        "ins_fire_load": ins_hor * ins_rho / 1e6,
        "hrr_inv": 1.0 / max(hrr_norm, 1e-6),
        "hrr_mesh": hrr_norm * mesh_norm,
        "hrr_mw": hrrpua * fire_area / 1000.0,
    }


def v4_feature_vector(fds_path, hrr_norm, mesh_norm):
    """Normalised v4 feature vector (list of N_V4_FEATURES floats)."""
    raw = extract_v4_raw(fds_path, hrr_norm, mesh_norm)
    out = []
    for name in V4_FEATURES:
        lo, hi = V4_NORM_RANGES[name]
        out.append((raw[name] - lo) / (hi - lo) if hi > lo else 0.0)
    return out


def build_params_v4(params, meta):
    """Append v4 features to the baseline (N, 16) params array -> (N, 16+K)."""
    extra = np.zeros((len(meta), N_V4_FEATURES), dtype=np.float32)
    for i, m in enumerate(meta):
        hrr_norm = params[i, 1]
        mesh_norm = params[i, 2]
        if m.get("fds_file"):
            extra[i] = v4_feature_vector(m["fds_file"], float(hrr_norm),
                                         float(mesh_norm))
        else:
            extra[i, V4_FEATURES.index("hrr_inv")] = (
                (1.0 / max(float(hrr_norm), 1e-6) - 0.5) / 1.3)
            extra[i, V4_FEATURES.index("hrr_mesh")] = (
                (float(hrr_norm) * float(mesh_norm) - 0.4) / 0.6)
            extra[i, V4_FEATURES.index("hrr_mw")] = (
                float(hrr_norm) * HRRPUA_REF * DEFAULT_FIRE_AREA / 1000.0
                / HRR_MW_SCALE)
    return np.concatenate([params, extra], axis=1)
