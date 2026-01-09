import os
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange, tqdm
from datetime import datetime

import tcm_piv as piv

from tcm_utils.cvd_check import set_cvd_friendly_colors
from tcm_utils.cough_model import CoughModel
from tcm_utils.camera_calibration import run_calibration, ensure_calibration_metadata
from tcm_utils.io_utils import load_json_key
from tcm_utils.file_dialogs import ask_directory, find_repo_root

print("\n\nStarting PIV analysis...")

# Set CVD-friendly colors
set_cvd_friendly_colors()

# Set paths (hard code for now)
repo_root = find_repo_root(Path(__file__))
data_root = repo_root / "examples" / "two_frames"
data_root.mkdir(parents=True, exist_ok=True)
output_root = repo_root / "examples" / "two_frames" / "output"
output_root.mkdir(parents=True, exist_ok=True)

# Handle calibration
calibration_metadata_path = ensure_calibration_metadata(
    input_path=data_root, distance_mm=1.0, output_dir=data_root)
calibration = load_json_key(calibration_metadata_path, "calibration")
print(f"Using calibration scale: {calibration['scale_mm_per_px']:.4f} mm/px")
