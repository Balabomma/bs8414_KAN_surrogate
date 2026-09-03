"""
Data loading and preprocessing for BS8414 FDS surrogate model.

Automatically scans simulation folders to find _devc.csv and .fds files.
Uses deterministic hash-based splitting so each simulation always lands in the
same set (train/valid/test) regardless of how many simulations are added later.

Supports two naming conventions:
  1. Parametric: DCLG_Test_1_PE_PIR_HRR1333_M010_0_1
  2. Legacy:     DCLG_Test_1_new_version02_0_08, DCLG_Test_3_FRPE_PIR_0_09, etc.
"""
import os
import re
import glob
import hashlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from config import (
    SIMS_DIR, DEVC_DIR, FDS_DIR, CLADDING_SYSTEMS, HRR_LEVELS, MESH_SIZES,
    N_TIMESTEPS, N_SENSORS, TRAIN_RATIO, VALID_RATIO, TEST_RATIO,
    MATERIAL_FEATURES, N_MATERIAL_FEATURES, N_INPUT_PARAMS,
)


# ──────────────────────────────────────────────────────────────────────
# Parameter extraction from folder names / FDS files
# ──────────────────────────────────────────────────────────────────────

def parse_chid_parametric(chid):
    """Parse parametric naming: DCLG_Test_7_FRPE_Phenolic_HRR2333_M010_0_1"""
    mesh_match = re.search(r"_(M\d{3})_", chid)
    hrr_match = re.search(r"_HRR(\d+)_", chid)
    clad_match = re.search(r"DCLG_(Test_\d+_.+?)_HRR", chid)

    if mesh_match and hrr_match and clad_match:
        return clad_match.group(1), int(hrr_match.group(1)), mesh_match.group(1)
    return None, None, None


def parse_chid_legacy(chid):
    """Parse legacy naming: DCLG_Test_1_new_version02_0_08 or DCLG_Test_3_FRPE_PIR_0_09

    These all have HRRPUA=2333.3 in their FDS files.
    Mesh size is the trailing number (0_08=0.08m, 0_09=0.09m, 0_1=0.1m, 08=0.08m).
    """
    # Map legacy folder names to cladding systems
    legacy_patterns = {
        r"DCLG_Test_1_new_version02": "Test_1_PE_PIR",
        r"DCLG_Test_3_FRPE_PIR(?!_HRR)": "Test_3_FRPE_PIR",
        r"DCLG_Test_5_LCM_PIR(?!_HRR)": "Test_5_LCM_PIR",
        r"DCLG_Test_7_FRPE_Phenolic(?!_HRR)": "Test_7_FRPE_Phenolic",
    }

    cladding = None
    for pattern, clad_name in legacy_patterns.items():
        if re.match(pattern, chid):
            cladding = clad_name
            break

    if cladding is None:
        return None, None, None

    # Extract mesh size from trailing part
    # Patterns: _0_08 -> 0.08, _0_09 -> 0.09, _0_1 -> 0.1, _08 -> 0.08
    mesh_match = re.search(r"_0_(\d+)$", chid)
    if mesh_match:
        mesh_val = float(f"0.{mesh_match.group(1)}")
    else:
        mesh_match = re.search(r"_(\d{2})$", chid)
        if mesh_match:
            mesh_val = float(f"0.{mesh_match.group(1)}")
        else:
            return None, None, None

    # Map to mesh key
    mesh_key = None
    for key, val in MESH_SIZES.items():
        if abs(val - mesh_val) < 0.001:
            mesh_key = key
            break

    if mesh_key is None:
        return None, None, None

    # All legacy sims use HRRPUA=2333
    hrr = 2333

    return cladding, hrr, mesh_key


def parse_chid(chid):
    """Parse any CHID format. Tries parametric first, then legacy."""
    cladding, hrr, mesh = parse_chid_parametric(chid)
    if cladding is not None:
        return cladding, hrr, mesh
    return parse_chid_legacy(chid)


def extract_hrrpua_from_fds(fds_path):
    """Read HRRPUA value directly from FDS file as fallback."""
    try:
        with open(fds_path, "r") as f:
            content = f.read()
        match = re.search(r"HRRPUA\s*=\s*([\d.]+)", content)
        if match:
            return int(float(match.group(1)))
    except Exception:
        pass
    return None


def _extract_matl_props(content, matl_id_patterns):
    """Extract properties from the first matching MATL block."""
    for pattern in matl_id_patterns:
        block_match = re.search(
            rf"&MATL\s+ID\s*=\s*'{pattern}'.*?/", content, re.DOTALL | re.IGNORECASE
        )
        if block_match:
            block = block_match.group(0)
            props = {}
            for key in ["SPECIFIC_HEAT", "CONDUCTIVITY", "DENSITY", "EMISSIVITY",
                         "HEAT_OF_REACTION", "REFERENCE_TEMPERATURE"]:
                m = re.search(rf"{key}\s*=\s*([\d.Ee+-]+)", block)
                props[key] = float(m.group(1)) if m else 0.0
            props["reactive"] = 1.0 if "N_REACTIONS" in block else 0.0
            return props
    return None


def _extract_reac_props(content, fuel_patterns):
    """Extract combustion reaction properties."""
    for pattern in fuel_patterns:
        block_match = re.search(
            rf"&REAC.*?FUEL\s*=\s*'{pattern}'.*?/", content, re.DOTALL | re.IGNORECASE
        )
        if block_match:
            block = block_match.group(0)
            props = {}
            for key in ["HEAT_OF_COMBUSTION", "CO_YIELD", "SOOT_YIELD"]:
                m = re.search(rf"{key}\s*=\s*([\d.Ee+-]+)", block)
                props[key] = float(m.group(1)) if m else 0.0
            return props
    return None


def extract_material_properties(fds_path):
    """Parse FDS file and return material property feature vector.

    Returns: list of 13 floats matching MATERIAL_FEATURES in config.py
        [core_cp, core_k, core_rho, core_HoR, core_Tref, core_reactive,
         ins_cp, ins_k, ins_rho, ins_HoR, ins_Tref,
         reac_HoC, reac_soot]
    """
    try:
        with open(fds_path, "r") as f:
            content = f.read()
    except Exception:
        return None

    # Extract core material (cladding panel core)
    core = _extract_matl_props(content, ["Core_MATERIAL", "Core_FRPE", "Core_PE"])
    if core is None:
        core = {"SPECIFIC_HEAT": 0, "CONDUCTIVITY": 0, "DENSITY": 0,
                "HEAT_OF_REACTION": 0, "REFERENCE_TEMPERATURE": 0, "reactive": 0}

    # Extract insulation material
    ins = _extract_matl_props(content, ["Insulation_VIRGIN", "Phenolic_VIRGIN",
                                         "PIR_VIRGIN", "Insulation"])
    if ins is None:
        ins = {"SPECIFIC_HEAT": 0, "CONDUCTIVITY": 0, "DENSITY": 0,
               "HEAT_OF_REACTION": 0, "REFERENCE_TEMPERATURE": 0}

    # Extract core combustion reaction (not insulation reaction)
    reac = _extract_reac_props(content, ["ETHYLENE", "Phenolic_GAS"])
    if reac is None:
        reac = {"HEAT_OF_COMBUSTION": 0, "SOOT_YIELD": 0}

    features = [
        core["SPECIFIC_HEAT"],
        core["CONDUCTIVITY"],
        core["DENSITY"],
        core["HEAT_OF_REACTION"],
        core["REFERENCE_TEMPERATURE"],
        core["reactive"],
        ins["SPECIFIC_HEAT"],
        ins["CONDUCTIVITY"],
        ins["DENSITY"],
        ins["HEAT_OF_REACTION"],
        ins["REFERENCE_TEMPERATURE"],
        reac["HEAT_OF_COMBUSTION"],
        reac["SOOT_YIELD"],
    ]

    return features


# Normalization ranges for material features (from observed FDS data)
MATERIAL_NORM_RANGES = {
    "core_specific_heat": (0, 4.0),
    "core_conductivity": (0, 0.5),
    "core_density": (0, 2000.0),
    "core_heat_of_reaction": (0, 3000.0),
    "core_ref_temperature": (0, 600.0),
    "core_is_reactive": (0, 1.0),
    "ins_specific_heat": (0, 2.0),
    "ins_conductivity": (0, 0.1),
    "ins_density": (0, 50.0),
    "ins_heat_of_reaction": (0, 2000.0),
    "ins_ref_temperature": (0, 500.0),
    "reac_heat_of_combustion": (0, 50000.0),
    "reac_soot_yield": (0, 0.2),
}


def normalize_material_features(raw_features):
    """Normalize material features to [0, 1] range."""
    normalized = []
    for val, feat_name in zip(raw_features, MATERIAL_FEATURES):
        lo, hi = MATERIAL_NORM_RANGES[feat_name]
        if hi > lo:
            normalized.append((val - lo) / (hi - lo))
        else:
            normalized.append(0.0)
    return normalized


# ──────────────────────────────────────────────────────────────────────
# Simulation discovery
# ──────────────────────────────────────────────────────────────────────

def scan_simulation_folders(sims_dir):
    """Scan per-simulation folders and locate _devc.csv and .fds files."""
    simulations = []

    if not os.path.isdir(sims_dir):
        print(f"  Simulations directory not found: {sims_dir}")
        return simulations

    for folder_name in sorted(os.listdir(sims_dir)):
        folder_path = os.path.join(sims_dir, folder_name)

        if not os.path.isdir(folder_path) or folder_name.startswith("_"):
            continue

        cladding, hrr, mesh = parse_chid(folder_name)
        if cladding is None or hrr is None or mesh is None:
            continue

        devc_files = glob.glob(os.path.join(folder_path, "*_devc.csv"))
        devc_csv = devc_files[0] if devc_files else None

        fds_files = glob.glob(os.path.join(folder_path, "*.fds"))
        fds_file = fds_files[0] if fds_files else None

        # If HRR wasn't in folder name, try reading from FDS file
        if hrr is None and fds_file:
            hrr = extract_hrrpua_from_fds(fds_file)

        simulations.append({
            "chid": folder_name,
            "folder": folder_path,
            "devc_csv": devc_csv,
            "fds_file": fds_file,
            "cladding": cladding,
            "hrr": hrr,
            "mesh": mesh,
        })

    return simulations


def scan_flat_folders(devc_dir, fds_dir):
    """Fallback: scan legacy flat directories for DEVC CSVs."""
    simulations = []

    if not os.path.isdir(devc_dir):
        return simulations

    csv_files = sorted(glob.glob(os.path.join(devc_dir, "*_devc.csv")))

    for csv_path in csv_files:
        filename = os.path.basename(csv_path)
        chid = filename.replace("_devc.csv", "")

        cladding, hrr, mesh = parse_chid(chid)
        if cladding is None or hrr is None or mesh is None:
            continue

        fds_path = os.path.join(fds_dir, f"{chid}.fds") if fds_dir else None
        if fds_path and not os.path.isfile(fds_path):
            fds_path = None

        simulations.append({
            "chid": chid,
            "folder": devc_dir,
            "devc_csv": csv_path,
            "fds_file": fds_path,
            "cladding": cladding,
            "hrr": hrr,
            "mesh": mesh,
        })

    return simulations


def discover_simulations():
    """Auto-discover simulations from all available sources."""
    print("  Scanning for simulations...")
    all_sims = {}

    folder_sims = scan_simulation_folders(SIMS_DIR)
    if folder_sims:
        with_csv = [s for s in folder_sims if s["devc_csv"] is not None]
        print(f"  [training_data/] {len(folder_sims)} folders, {len(with_csv)} with DEVC CSV")
        for s in folder_sims:
            all_sims[s["chid"]] = s

    flat_sims = scan_flat_folders(DEVC_DIR, FDS_DIR)
    if flat_sims:
        new_count = 0
        for s in flat_sims:
            if s["chid"] not in all_sims:
                all_sims[s["chid"]] = s
                new_count += 1
            elif all_sims[s["chid"]]["devc_csv"] is None and s["devc_csv"] is not None:
                all_sims[s["chid"]]["devc_csv"] = s["devc_csv"]
                new_count += 1
        if new_count > 0:
            print(f"  [devc_csv/]      {new_count} additional from flat layout")

    sims = sorted(all_sims.values(), key=lambda s: s["chid"])
    with_csv = [s for s in sims if s["devc_csv"] is not None]
    print(f"  Total: {len(sims)} discovered, {len(with_csv)} with output data")

    return sims


# ──────────────────────────────────────────────────────────────────────
# CSV loading
# ──────────────────────────────────────────────────────────────────────

def load_devc_csv(filepath):
    """Load a DEVC CSV file and return time array and temperature matrix."""
    df = pd.read_csv(filepath, skiprows=1)
    time = df.iloc[:, 0].values
    temperatures = df.iloc[:, 1:].values
    columns = list(df.columns[1:])
    return time, temperatures, columns


# ──────────────────────────────────────────────────────────────────────
# Dataset building
# ──────────────────────────────────────────────────────────────────────

def build_dataset(include_partial=False, min_timesteps=30):
    """Discover and load simulations. Optionally include partial sims.

    Args:
        include_partial: If True, include incomplete sims (padded + masked)
        min_timesteps: Minimum timesteps to include a partial sim

    Returns:
        params: np.array (n_samples, N_INPUT_PARAMS)
        outputs: np.array (n_samples, n_timesteps, n_sensors) - padded to N_TIMESTEPS
        masks: np.array (n_samples, n_timesteps) - 1 where data exists, 0 where padded
        meta: list of dicts
        sensor_names: list of sensor column names
    """
    all_sims = discover_simulations()

    params_list = []
    outputs_list = []
    masks_list = []
    meta_list = []
    sensor_names = None
    skipped_no_csv = 0
    skipped_too_short = 0
    skipped_bad_params = 0
    n_complete = 0
    n_partial = 0

    for sim in all_sims:
        if sim["devc_csv"] is None:
            skipped_no_csv += 1
            continue

        if sim["cladding"] not in CLADDING_SYSTEMS:
            skipped_bad_params += 1
            continue

        if sim["mesh"] not in MESH_SIZES:
            skipped_bad_params += 1
            continue

        time, temps, columns = load_devc_csv(sim["devc_csv"])

        if temps.shape[1] != N_SENSORS:
            skipped_bad_params += 1
            continue

        n_steps = len(time)
        is_complete = (n_steps == N_TIMESTEPS)

        if not is_complete:
            if not include_partial or n_steps < min_timesteps:
                if n_steps < min_timesteps:
                    skipped_too_short += 1
                else:
                    skipped_too_short += 1
                continue
            # Pad partial sim to N_TIMESTEPS
            padded = np.zeros((N_TIMESTEPS, N_SENSORS), dtype=np.float32)
            padded[:n_steps, :] = temps[:n_steps, :]
            mask = np.zeros(N_TIMESTEPS, dtype=np.float32)
            mask[:n_steps] = 1.0
            temps = padded
            n_partial += 1
            print(f"  Partial: {sim['chid']} ({n_steps}/{N_TIMESTEPS} timesteps)")
        else:
            mask = np.ones(N_TIMESTEPS, dtype=np.float32)
            n_complete += 1

        if sensor_names is None:
            sensor_names = columns

        cladding_id = CLADDING_SYSTEMS[sim["cladding"]]
        hrr_norm = sim["hrr"] / max(HRR_LEVELS)
        mesh_norm = MESH_SIZES[sim["mesh"]] / max(MESH_SIZES.values())

        mat_features = [0.0] * N_MATERIAL_FEATURES
        if sim["fds_file"] and os.path.isfile(sim["fds_file"]):
            raw = extract_material_properties(sim["fds_file"])
            if raw is not None:
                mat_features = normalize_material_features(raw)

        params_list.append([cladding_id, hrr_norm, mesh_norm] + mat_features)
        outputs_list.append(temps)
        masks_list.append(mask)
        meta_list.append({
            "chid": sim["chid"],
            "folder": sim["folder"],
            "devc_csv": sim["devc_csv"],
            "fds_file": sim["fds_file"],
            "cladding": sim["cladding"],
            "hrr": sim["hrr"],
            "mesh": sim["mesh"],
            "material_features": mat_features,
            "n_timesteps": n_steps,
            "is_complete": is_complete,
        })

    params = np.array(params_list, dtype=np.float32)
    outputs = np.array(outputs_list, dtype=np.float32)
    masks = np.array(masks_list, dtype=np.float32)

    print(f"\n  Summary:")
    print(f"    Complete:   {n_complete} simulations")
    print(f"    Partial:    {n_partial} simulations (padded + masked)")
    print(f"    Total:      {n_complete + n_partial} usable")
    print(f"    No output:  {skipped_no_csv}")
    print(f"    Too short:  {skipped_too_short} (< {min_timesteps} timesteps)")
    print(f"    Bad params: {skipped_bad_params}")
    print(f"    Input:  {params.shape}")
    print(f"    Output: {outputs.shape}")

    return params, outputs, masks, meta_list, sensor_names


# ──────────────────────────────────────────────────────────────────────
# Deterministic hash-based splitting
# ──────────────────────────────────────────────────────────────────────

def assign_split(chid, train_ratio=TRAIN_RATIO, valid_ratio=VALID_RATIO):
    """Deterministically assign a simulation to train/valid/test using its CHID hash.

    The hash of the CHID string produces a stable number in [0, 1).
    This ensures the same simulation always goes to the same split,
    even when new simulations are added later.

    Buckets:
        [0, train_ratio)                         -> train
        [train_ratio, train_ratio + valid_ratio)  -> valid
        [train_ratio + valid_ratio, 1.0)          -> test
    """
    h = hashlib.md5(chid.encode()).hexdigest()
    bucket = int(h[:8], 16) / 0xFFFFFFFF  # Stable float in [0, 1)

    if bucket < train_ratio:
        return "train"
    elif bucket < train_ratio + valid_ratio:
        return "valid"
    else:
        return "test"


class FDSSurrogateDataset(Dataset):
    """PyTorch dataset with mask support for partial simulations."""

    def __init__(self, params, outputs, masks, time_array):
        self.params = torch.FloatTensor(params)
        self.outputs = torch.FloatTensor(outputs)
        self.masks = torch.FloatTensor(masks)
        self.time_array = torch.FloatTensor(time_array)

    def __len__(self):
        return len(self.params)

    def __getitem__(self, idx):
        return self.params[idx], self.outputs[idx], self.masks[idx], self.time_array


def prepare_data_splits(params, outputs, masks, meta):
    """Split data into train/valid/test using deterministic hash-based assignment."""
    n_samples = len(params)
    time_array = np.linspace(0, 1, N_TIMESTEPS).astype(np.float32)

    # Normalize temperature outputs
    orig_shape = outputs.shape
    flat_outputs = outputs.reshape(-1, N_SENSORS)

    output_scaler = StandardScaler()
    flat_scaled = output_scaler.fit_transform(flat_outputs)
    outputs_scaled = flat_scaled.reshape(orig_shape)

    # Deterministic split assignment
    train_idx, valid_idx, test_idx = [], [], []
    for i, m in enumerate(meta):
        split = assign_split(m["chid"])
        if split == "train":
            train_idx.append(i)
        elif split == "valid":
            valid_idx.append(i)
        else:
            test_idx.append(i)

    train_idx = np.array(train_idx)
    valid_idx = np.array(valid_idx)
    test_idx = np.array(test_idx)

    # Safety: ensure at least 1 sample per split
    if len(valid_idx) == 0 and len(train_idx) > 2:
        valid_idx = np.array([train_idx[-1]])
        train_idx = train_idx[:-1]
    if len(test_idx) == 0 and len(train_idx) > 2:
        test_idx = np.array([train_idx[-1]])
        train_idx = train_idx[:-1]

    train_meta = [meta[i] for i in train_idx]
    valid_meta = [meta[i] for i in valid_idx]
    test_meta = [meta[i] for i in test_idx]

    print(f"\nData splits (deterministic, hash-based):")
    print(f"  Train: {len(train_idx)} samples")
    for m in train_meta:
        print(f"    - {m['cladding']} HRR{m['hrr']} {m['mesh']}  [{m['chid']}]")
    print(f"  Valid: {len(valid_idx)} samples")
    for m in valid_meta:
        print(f"    - {m['cladding']} HRR{m['hrr']} {m['mesh']}  [{m['chid']}]")
    print(f"  Test:  {len(test_idx)} samples")
    for m in test_meta:
        print(f"    - {m['cladding']} HRR{m['hrr']} {m['mesh']}  [{m['chid']}]")

    train_dataset = FDSSurrogateDataset(
        params[train_idx], outputs_scaled[train_idx], masks[train_idx], time_array
    )
    valid_dataset = FDSSurrogateDataset(
        params[valid_idx], outputs_scaled[valid_idx], masks[valid_idx], time_array
    ) if len(valid_idx) > 0 else None
    test_dataset = FDSSurrogateDataset(
        params[test_idx], outputs_scaled[test_idx], masks[test_idx], time_array
    ) if len(test_idx) > 0 else None

    split_info = {
        "train_idx": train_idx,
        "valid_idx": valid_idx,
        "test_idx": test_idx,
        "train_meta": train_meta,
        "valid_meta": valid_meta,
        "test_meta": test_meta,
    }

    return train_dataset, valid_dataset, test_dataset, output_scaler, split_info, time_array


def prepare_kfold_splits(params, outputs, masks, meta, n_folds=5):
    """Create K-Fold cross-validation splits.

    Every simulation is tested exactly once across all folds.
    Returns list of (train_idx, valid_idx, test_idx) per fold.
    """
    from sklearn.model_selection import KFold

    time_array = np.linspace(0, 1, N_TIMESTEPS).astype(np.float32)
    n_samples = len(params)

    # Fit scaler on ALL data
    output_scaler = StandardScaler()
    flat = outputs.reshape(-1, N_SENSORS)
    flat_scaled = output_scaler.fit_transform(flat)
    outputs_scaled = flat_scaled.reshape(outputs.shape)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    folds = []

    for fold_i, (train_val_idx, test_idx) in enumerate(kf.split(np.arange(n_samples))):
        # Split train_val into train (80%) and valid (20%)
        n_val = max(1, int(len(train_val_idx) * 0.2))
        valid_idx = train_val_idx[:n_val]
        train_idx = train_val_idx[n_val:]

        train_ds = FDSSurrogateDataset(
            params[train_idx], outputs_scaled[train_idx], masks[train_idx], time_array
        )
        valid_ds = FDSSurrogateDataset(
            params[valid_idx], outputs_scaled[valid_idx], masks[valid_idx], time_array
        )
        test_ds = FDSSurrogateDataset(
            params[test_idx], outputs_scaled[test_idx], masks[test_idx], time_array
        )

        fold_info = {
            "fold": fold_i + 1,
            "train_idx": train_idx, "valid_idx": valid_idx, "test_idx": test_idx,
            "train_meta": [meta[i] for i in train_idx],
            "valid_meta": [meta[i] for i in valid_idx],
            "test_meta": [meta[i] for i in test_idx],
        }

        folds.append((train_ds, valid_ds, test_ds, fold_info))
        print(f"  Fold {fold_i+1}: Train={len(train_idx)} Valid={len(valid_idx)} Test={len(test_idx)}")

    return folds, output_scaler, time_array


if __name__ == "__main__":
    print("=" * 70)
    print("  BS8414 FDS Surrogate Model — Data Discovery")
    print("=" * 70)

    params, outputs, masks, meta, sensor_names = build_dataset(include_partial=True)

    if sensor_names:
        print(f"\nSensor names: {len(sensor_names)}")

    print(f"\nAll simulations:")
    print(f"  {'CHID':<55} {'Cladding':<22} {'HRR':>5} {'Mesh':>5} {'Steps':>6}")
    print(f"  {'-'*55} {'-'*22} {'-'*5} {'-'*5} {'-'*6}")
    for m in meta:
        status = "FULL" if m["is_complete"] else f"PARTIAL({m['n_timesteps']})"
        print(f"  {m['chid']:<55} {m['cladding']:<22} {m['hrr']:>5} {m['mesh']:>5} {status:>12}")
