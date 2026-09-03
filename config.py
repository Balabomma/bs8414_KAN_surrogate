"""Configuration for BS8414 KAN-Attention-LSTM Surrogate Model."""
import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
SIMS_DIR = os.path.join(DATA_DIR, "training_data")
DEVC_DIR = os.path.join(DATA_DIR, "devc_csv")
FDS_DIR = os.path.join(DATA_DIR, "fds_scripts")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
MODEL_DIR = os.path.join(PROJECT_DIR, "models")

TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO = 0.15

CLADDING_SYSTEMS = {
    "Test_1_PE_PIR": 0, "Test_3_FRPE_PIR": 1,
    "Test_5_LCM_PIR": 2, "Test_7_FRPE_Phenolic": 3,
}
HRR_LEVELS = [1333, 1667, 2000, 2100, 2333]
MESH_SIZES = {"M008": 0.08, "M009": 0.09, "M010": 0.10}

MATERIAL_FEATURES = [
    "core_specific_heat", "core_conductivity", "core_density",
    "core_heat_of_reaction", "core_ref_temperature", "core_is_reactive",
    "ins_specific_heat", "ins_conductivity", "ins_density",
    "ins_heat_of_reaction", "ins_ref_temperature",
    "reac_heat_of_combustion", "reac_soot_yield",
]
N_MATERIAL_FEATURES = len(MATERIAL_FEATURES)
N_INPUT_PARAMS = 3 + N_MATERIAL_FEATURES  # 16

T_END = 1800.0
DT_DEVC = 10.0
N_TIMESTEPS = 181
N_SENSORS = 24

KEY_THERMOCOUPLES = {
    "External_LV2_main02(1029)": "TC 1029 - External LV2 Main",
    "External_LV1_main02(1003)": "TC 1003 - External LV1 Main",
}

DEVICE = "cuda"
