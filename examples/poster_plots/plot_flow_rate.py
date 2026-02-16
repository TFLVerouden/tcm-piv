"""Quick-and-dirty flow-rate plotting script (no CLI).

Hardcoded to load four flow_rate.csv chunks from the DefaultFlowCurve1000mBar
dataset and concatenate them in time.

Edits you likely want to make:
- DATA_RUN_DIRS: point at different runs/chunks
- MODEL_*: change Gupta model parameters
- OUT_PDF: output filename

Notes:
- Uses the Matplotlib style from tcm_utils.plot_style (tcm-poster).
- No plot title, figure size (4, 3).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from tcm_utils.cough_model import CoughModel
from tcm_utils.plot_style import (
    add_label,
    append_unit_to_last_ticklabel,
    raise_axis_frame,
    set_grid,
    set_ticks_every,
    use_tcm_poster_style,
)


def _read_flow_rate_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read time_s and flow_rate_L_s from a flow_rate.csv."""

    # Expected header:
    # pair_index,time_s,flow_rate_m3_s,flow_rate_L_s
    arr = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )

    if "time_s" not in arr.dtype.names or "flow_rate_L_s" not in arr.dtype.names:
        raise ValueError(
            f"CSV missing required columns. Found: {arr.dtype.names}. "
            "Need at least: time_s, flow_rate_L_s"
        )

    time_s = np.asarray(arr["time_s"], dtype=float)
    flow_lps = np.asarray(arr["flow_rate_L_s"], dtype=float)
    return time_s, flow_lps


def _clean_finite(time_s: np.ndarray, flow_lps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mask = np.isfinite(time_s) & np.isfinite(flow_lps)
    return time_s[mask], flow_lps[mask]


def _estimate_dt(time_s: np.ndarray) -> float:
    if len(time_s) < 2:
        return 0.0
    diffs = np.diff(time_s)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return 0.0
    return float(np.median(diffs))


def _concat_time_series(pieces: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate (time_s, y) pieces, auto-offsetting time if needed.

    If a subsequent piece restarts its time axis (e.g. starts near 0 or starts
    before/at the previous end), we shift it forward so it follows directly
    after the previous piece.
    """

    if not pieces:
        return np.array([], dtype=float), np.array([], dtype=float)

    t_all: list[np.ndarray] = []
    y_all: list[np.ndarray] = []

    t_prev_end: float | None = None
    dt_prev: float = 0.0

    for time_s, y in pieces:
        time_s, y = _clean_finite(np.asarray(
            time_s, dtype=float), np.asarray(y, dtype=float))
        if len(time_s) == 0:
            continue

        if t_prev_end is None:
            t_shifted = time_s
        else:
            t0 = float(time_s[0])
            eps = max(1e-12, 0.25 * max(dt_prev, _estimate_dt(time_s)))
            if t0 <= (t_prev_end + eps):
                # Shift so this piece starts just after the previous end.
                dt_use = dt_prev if dt_prev > 0 else _estimate_dt(time_s)
                offset = t_prev_end + dt_use - t0
                t_shifted = time_s + offset
            else:
                # Already strictly after previous; assume it has absolute time.
                t_shifted = time_s

        t_all.append(t_shifted)
        y_all.append(y)

        t_prev_end = float(t_shifted[-1])
        dt_prev = _estimate_dt(t_shifted)

    if not t_all:
        return np.array([], dtype=float), np.array([], dtype=float)

    return np.concatenate(t_all), np.concatenate(y_all)


def _auto_tick_step(span: float) -> float:
    """Pick a reasonable tick step for quick plots."""

    if span <= 0:
        return 1.0
    if span <= 0.01:
        return 0.002
    if span <= 0.05:
        return 0.01
    if span <= 0.2:
        return 0.05
    if span <= 0.6:
        return 0.1
    if span <= 2.0:
        return 0.25
    if span <= 6:
        return 1.0
    if span <= 20:
        return 2.0
    if span <= 60:
        return 5.0
    if span <= 120:
        return 10.0
    if span <= 300:
        return 25.0
    return 50.0


def _time_tick_fmt(span_s: float) -> str:
    if span_s < 0.02:
        return "{x:.3f}"
    if span_s < 0.2:
        return "{x:.2f}"
    return "{x:.1f}"
# ========== EDIT THESE ==========


DATA_RUN_DIRS = [
    Path(
        "/Users/tommieverouden/Documents/Data/PIV/260105 Proportional valve tests/DefaultFlowCurve1000mBar/output/runs/260115_225112"
    ),
    Path(
        "/Users/tommieverouden/Documents/Data/PIV/260105 Proportional valve tests/DefaultFlowCurve1000mBar/output/runs/260115_230530"
    ),
    Path(
        "/Users/tommieverouden/Documents/Data/PIV/260105 Proportional valve tests/DefaultFlowCurve1000mBar/output/runs/260115_231422"
    ),
    Path(
        "/Users/tommieverouden/Documents/Data/PIV/260105 Proportional valve tests/DefaultFlowCurve1000mBar/output/runs/260115_232330"
    ),
]

OUT_PDF = Path("flow_rate_defaultflowcurve_1000mbar.pdf")

# Gupta model params (quick to edit)
MODEL_GENDER = "Male"
MODEL_WEIGHT_KG = 100.0
MODEL_HEIGHT_M = 2


# ========== LOAD + CONCAT DATA ==========

CSV_PATHS = [d / "flow_rate.csv" for d in DATA_RUN_DIRS]
missing = [p for p in CSV_PATHS if not p.is_file()]
if missing:
    raise FileNotFoundError("CSV not found: " + ", ".join(map(str, missing)))

pieces: list[tuple[np.ndarray, np.ndarray]] = []
for pth in CSV_PATHS:
    t_s, q_lps = _read_flow_rate_csv(pth)
    pieces.append((t_s, q_lps))

time_s, flow_lps = _concat_time_series(pieces)
if len(time_s) == 0:
    raise ValueError("No finite data across provided CSVs")


# ========== MODEL (GUPTA) ==========

cough = CoughModel.from_example("Feinstein")
cough2 = CoughModel.from_gupta(
    gender=str(MODEL_GENDER),
    weight_kg=float(MODEL_WEIGHT_KG),
    height_m=float(MODEL_HEIGHT_M),
)
t_model_s, q_model_lps = cough.flow(units="L/s")
t_model_s = np.asarray(t_model_s, dtype=float)
q_model_lps = np.asarray(q_model_lps, dtype=float)

t_model_s2, q_model_lps2 = cough2.flow(time_s=time_s, units="L/s")
t_model_s2 = np.asarray(t_model_s2, dtype=float)
q_model_lps2 = np.asarray(q_model_lps2, dtype=float)


# ========== PLOT ==========

use_tcm_poster_style()

fig, ax = plt.subplots(1, 1, figsize=(4.0, 3.0), constrained_layout=True)

# Match the demo-1 vibe: no title, minimal extras.
set_grid(ax, mode="horizontal", on=False)

# ax.plot(
#     t_model_s2,
#     q_model_lps2,
#     linestyle="--",
#     color="#a9cba0",
#     linewidth=3.0,
# )
ax.plot(time_s, flow_lps, color="#2ba02b", linewidth=4)
ax.plot(
    t_model_s,
    q_model_lps,
    linestyle="",
    marker="",
    markersize=6,
    markeredgewidth=3,
    linewidth=4,
    color="#a9cba0"
)

ax.set_ylabel("Flow rate (L/s)")

# Nice limits, similar to demo 1
t_span = float(np.nanmax(time_s))
ax.set_xlim(-0.02, 0.45)
y_max = float(np.nanmax([np.nanmax(flow_lps), np.nanmax(q_model_lps)]))
ax.set_ylim(0.0, max(1.0, y_max * 1.05))

# In-plot labels (legend replacement)
# add_label(ax, "cough machine", xy=(0.55, 0.8),
#           coord_system="axes", ha="left", va="top")
# add_label(ax, "model", xy=(0.98, 0.95),
#           coord_system="axes", ha="right", va="top")

# Ticks: light-touch, auto-ish
set_ticks_every(ax, axis="x", step=0.2)
set_ticks_every(ax, axis="y", step=2)
append_unit_to_last_ticklabel(
    ax, axis="x", unit="s", fmt=_time_tick_fmt(t_span))

raise_axis_frame(ax)

OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PDF, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {OUT_PDF}")
