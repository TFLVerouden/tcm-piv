"""Quick-and-dirty velocity profile plot (no CLI, single-use).

Reads one velocity_final.csv, takes the median profile between 40–60 ms as a
function of distance from the channel center, and shades the IQR.

Edit the constants below and run:
	/path/to/.venv/bin/python plot_profile.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from tcm_utils.plot_style import (
    add_label,
    raise_axis_frame,
    set_grid,
    set_ticks_every,
    use_tcm_poster_style,
)


# ===================== EDIT THESE =====================

CSV_PATH = Path(
    "/Users/tommieverouden/Documents/Data/PIV/260105 Proportional valve tests/DefaultFlowCurve1000mBar/output/runs/260115_225112/velocity_final.csv"
)

OUT_PDF = Path("velocity_profile_40_60ms.pdf")

T0_S = 0.040
T1_S = 0.060

# Colors
LINE_COLOR = "#d62727"
FILL_COLOR = "#e5a4a0"


# ===================== LOAD =====================

if not CSV_PATH.is_file():
    raise FileNotFoundError(f"Not found: {CSV_PATH}")

# Columns:
# pair_index,win_y,win_x,time_s,vy_m_s,vx_m_s
# Usecols:   0        1     2     3     4      5
usecols = (2, 3, 4)
a = np.genfromtxt(CSV_PATH, delimiter=",", skip_header=1, usecols=usecols)
if a.ndim == 1:
    a = a.reshape(1, -1)

win_x = np.asarray(a[:, 0], dtype=float)
time_s = np.asarray(a[:, 1], dtype=float)
vel = np.asarray(a[:, 2], dtype=float)

m = (
    np.isfinite(win_x)
    & np.isfinite(win_x)
    & np.isfinite(time_s)
    & np.isfinite(vel)
    & (time_s >= T0_S)
    & (time_s <= T1_S)
)
win_x = win_x[m]
time_s = time_s[m]
vel = vel[m]


# ===================== PROFILE (per-window time median, then across-x IQR) =====================

# Interpret indices as integers (CSV stores them as floats)
win_x_i = np.asarray(np.rint(win_x), dtype=int)

# For each window x index, get the per-window time median velocity
unique_win_x_i = np.unique(win_x_i)
v_med = []
v_q1 = []
v_q3 = []
for xi in unique_win_x_i:
    m = win_x_i == xi
    v_med.append(np.nanmedian(vel[m]))
    v_q1.append(np.nanpercentile(vel[m], 25.0))
    v_q3.append(np.nanpercentile(vel[m], 75.0))
v_med = np.asarray(v_med, dtype=float)
v_q1 = np.asarray(v_q1, dtype=float)
v_q3 = np.asarray(v_q3, dtype=float)

dist = unique_win_x_i.astype(float) - 11.5


# ===================== PLOT =====================

use_tcm_poster_style()

fig, ax = plt.subplots(1, 1, figsize=(4.0, 3.0), constrained_layout=True)
set_grid(ax, mode="horizontal", on=True)

ax.fill_between(dist, v_q1, v_q3, color=FILL_COLOR, alpha=0.85, linewidth=0.0)
ax.plot(dist, v_med, color=LINE_COLOR, linewidth=3.0)

ax.set_ylabel(f"Velocity (m/s)")

raise_axis_frame(ax)

OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PDF, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {OUT_PDF}")
