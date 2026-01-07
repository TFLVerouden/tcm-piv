import getpass
import os

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation as ani
from scipy import signal as sig
from scipy.interpolate import make_smoothing_spline
from tqdm import tqdm

import piv_functions as piv


# Set experimental parameters
meas_name = '250624_1431_80ms_nozzlepress1bar_cough05bar'
dt = 1 / 40000  # [s] 

# Data processing settings
v_max = [15, 100]  # [m/s]
ds_fac = 4  # First pass downsampling factor
n_peaks1 = 10  # Number of peaks to find in first pass correlation map
n_peaks2 = 5
n_windows2 = (8, 1)  # Number of windows in second pass (rows, cols)

# File handling
current_dir = os.path.dirname(os.path.abspath(__file__))
cal_path = os.path.join(current_dir, "calibration",
                        "250624_calibration_PIV_500micron_res_std.txt")
user = getpass.getuser()
if user == "tommieverouden":
    data_path = os.path.join("/Volumes/Data/Data/250623 PIV/", meas_name)
elif user == "sikke":
    data_path = os.path.join("D:\\Experiments\\PIV\\", meas_name)

# Data saving settings
disp1_var_names = ['time', 'disp1_unf', 'int1_unf', 'n_corrs']
disp2_var_names = ['disp2', 'disp2_unf', 'int2_unf', 'centres']

# In the current directory, create a folder for processed data
# named the same as the final part of the data_path
proc_path = os.path.join(current_dir, 'processed', os.path.basename(data_path))

# Read calibration data
if not os.path.exists(cal_path):
    raise FileNotFoundError(f"Calibration file not found: {cal_path}")
res_avg, _ = np.loadtxt(cal_path)

# Convert max velocities to max displacements in px
d_max = np.array(v_max) * dt / res_avg  # m/s -> px/frame

# Load data
bckp1_loaded, loaded_vars = piv.backup("load", proc_path, "pass1.npz", disp1_var_names)

if bckp1_loaded:
    # Extract loaded variables using the same names as defined in disp1_var_names
    for var_name in disp1_var_names:
        globals()[var_name] = loaded_vars.get(var_name)
    print("Loaded existing backup data.")

bckp2_loaded, loaded_vars = piv.backup("load", proc_path, "pass2.npz", disp2_var_names)

if bckp2_loaded:
    # Extract loaded variables using the same names as defined in disp2_var_names
    for var_name in disp2_var_names:
        globals()[var_name] = loaded_vars.get(var_name)
    print("Loaded existing backup data.")


# Plot unfiltered peak intensities vs peak number for both passes
fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# First pass
axs[0].set_title("First Pass: Unfiltered Peak Intensities vs Peak Number")
for i in range(int1_unf.shape[0]):  # frames
    for j in range(int1_unf.shape[1]):  # windows (usually 1)
        for k in range(int1_unf.shape[2]):  # windows (usually 1)
            axs[0].plot(range(int1_unf.shape[3]), int1_unf[i, j, k, :], alpha=0.3)
axs[0].set_ylabel("Intensity")
axs[0].set_yscale('log')
axs[0].set_xlabel("Peak Number")

# Second pass
axs[1].set_title("Second Pass: Unfiltered Peak Intensities vs Peak Number")
for i in range(int2_unf.shape[0]):  # frames
    for j in range(int2_unf.shape[1]):  # windows rows
        for k in range(int2_unf.shape[2]):  # windows cols
            axs[1].plot(range(int2_unf.shape[3]), int2_unf[i, j, k, :], alpha=0.3)
axs[1].set_ylabel("Intensity")
axs[1].set_yscale('log')
axs[1].set_xlabel("Peak Number")

plt.tight_layout()

# Finally, show all figures
plt.show()
