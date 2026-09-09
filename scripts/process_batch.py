from __future__ import annotations
import tcm_piv.visualisation as viz
from tcm_utils.file_dialogs import ask_open_file

import csv
from pathlib import Path
import re

import numpy as np

from matplotlib import pyplot as plt

# Timing constants
TRIGGER_DELAY_MS = -10
OPENING_DELAY_MS = 15

# Set interval to calculate average over
START_BASE_MS = -5
END_BASE_MS = 5
START_STEP_MS = 55
END_STEP_MS = 95

# Set the factor by which the flow must rise above the base average to be considered a step response
STEP_START_THRESHOLD_FACTOR = 1.5

# Name of the per-run velocity file, expected in the same folder as flow_rate.csv
VELOCITY_FILENAME = "velocity_final.csv"

# Calibration scale because I accidentally did not output the window locations in PIV
SCALE_M_PER_PX = 2.2101482641127257e-05
WIN_POS_Y = np.array([28, 72, 117, 162, 206, 251, 296, 341, 385,
                      430, 475, 520, 564, 609, 654, 699, 743, 788, 833, 878], dtype=float)
WIN_POS_X = np.full_like(WIN_POS_Y, 448.0, dtype=float)
# NOTE: window positions are also "averaged"; some might have been shifted in previous passes

# Center and scale window positions
win_pos_x_mm = (WIN_POS_X - np.mean(WIN_POS_X)) * SCALE_M_PER_PX * 1000
win_pos_y_mm = WIN_POS_Y * SCALE_M_PER_PX * 1000

# ==============================================================================
# IMPORT AND SORT DATA FROM MANIFEST
# ==============================================================================
manifest_path = ask_open_file(
    key="process_batch_manifest",
    title="Select the PIV result CSV",
    filetypes=(("CSV files", "*.csv"), ("All files", "*.*")),
    default_dir=Path.cwd(),
)

# manifest_path = "/Volumes/Data/PIV/260820_piv/step_1-5bar/260831_152310_step_1-5bar_20-0mA/260903_223913_piv_result.csv"

if manifest_path is None:
    raise RuntimeError("No PIV result CSV selected; aborting.")

manifest_path = Path(manifest_path)
if not manifest_path.is_file():
    raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")

with manifest_path.open("r", newline="", encoding="utf-8") as fp:
    rows = list(csv.DictReader(fp))

if not rows:
    raise RuntimeError(f"No rows found in {manifest_path}")

top_dir_path = Path(rows[0].get("top_dir_path", "").strip())
batch_run_id = rows[0].get("batch_run_id", "").strip()

# If the top_dir_path contains the word "base", set pressure and current to 0
if "base" in top_dir_path.name:
    pressure = 0.0
    current = 0.0
else:
    # From the top_dir_path, obtain the pressure and the valve current
    pressure_match = re.search(r'(\d+-\d+)bar', top_dir_path.name)
    current_match = re.search(r'(\d+-\d+)mA', top_dir_path.name)
    if pressure_match is None or current_match is None:
        raise ValueError(
            f"Could not parse pressure/current from {top_dir_path.name}")
    pressure = float(pressure_match.group(1).replace('-', '.'))
    current = float(current_match.group(1).replace('-', '.'))

labels: list[str] = []
time_arrays: list[np.ndarray] = []
flow_arrays: list[np.ndarray] = []
flow_series: list[tuple[str, np.ndarray, np.ndarray]] = []
velocity_paths: list[Path] = []

for index, row in enumerate(rows, start=1):
    label = row.get("subfolder") or f"run_{index}"
    flow_csv_text = row.get("flow_rate_csv", "").strip()
    if flow_csv_text:
        flow_csv = Path(flow_csv_text)
    else:
        run_dir_text = row.get("run_dir", "").strip()
        if not run_dir_text:
            raise KeyError(
                f"Row {index} in {manifest_path} needs either flow_rate_csv or run_dir"
            )
        flow_csv = Path(run_dir_text) / "flow_rate.csv"

    if not flow_csv.is_file():
        raise FileNotFoundError(f"flow_rate.csv not found: {flow_csv}")

    # velocity_final.csv always lives alongside flow_rate.csv
    vel_csv = flow_csv.parent / VELOCITY_FILENAME
    if not vel_csv.is_file():
        raise FileNotFoundError(f"{VELOCITY_FILENAME} not found: {vel_csv}")
    velocity_paths.append(vel_csv)

    data = np.genfromtxt(flow_csv, delimiter=",", names=True)
    if data.size == 0:
        raise RuntimeError(f"No data found in {flow_csv}")
    if getattr(data, "ndim", 0) == 0:
        data = np.array([data], dtype=data.dtype)

    time_s = np.asarray(data["time_s"], dtype=float).reshape(-1)
    flow_lps = np.asarray(data["flow_rate_L_s"], dtype=float).reshape(-1)
    finite = np.isfinite(time_s) & np.isfinite(flow_lps)
    time_s = time_s[finite]
    flow_lps = flow_lps[finite]
    if time_s.size == 0:
        raise RuntimeError(f"No finite data found in {flow_csv}")

    # Subtract delays from the time array to account for trigger and opening delays
    time_s -= (TRIGGER_DELAY_MS + OPENING_DELAY_MS) / 1000

    labels.append(label)
    time_arrays.append(time_s)
    flow_arrays.append(flow_lps)
    flow_series.append((label, time_s, flow_lps))

output_dir = manifest_path.parent
stem = manifest_path.stem


# Find the run-log folder that belongs to this batch so pressure data can be
# paired with the flow-rate rows from the manifest.
def _find_run_logs_dir(manifest_path: Path, top_dir_path: Path) -> Path:
    """Find the run_logs directory near the current batch manifest."""

    search_roots = [
        manifest_path.parent,
        top_dir_path,
        top_dir_path.parent,
        manifest_path.parent.parent,
    ]
    seen: set[Path] = set()

    for root in search_roots:
        if not root:
            continue

        direct = root / "run_logs"
        if direct.is_dir():
            return direct

        for candidate in root.rglob("run_logs"):
            if candidate.is_dir() and candidate not in seen:
                seen.add(candidate)
                print(f"Found run_logs folder: {candidate}")
                return candidate

    raise FileNotFoundError(
        f"Could not find a run_logs folder near {manifest_path}"
    )


def _read_run_log(run_log_path: Path) -> tuple[
    int,
    int,
    np.ndarray,
    np.ndarray,
]:
    """Read a run log and return timing plus pressure samples."""

    run_nr: int | None = None
    trigger_t0_us: int | None = None
    header_found = False

    time_us: list[int] = []
    press_bar: list[float] = []

    with run_log_path.open("r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue

            key = row[0].strip()
            if key == "run_nr" and len(row) > 1:
                run_nr = int(row[1])
                continue

            if key == "trigger_t0_us" and len(row) > 1:
                trigger_t0_us = int(row[1])
                continue

            if key == "time_us":
                header_found = True
                continue

            if not header_found or len(row) < 4:
                continue

            time_us.append(int(row[0]))
            press_bar.append(float(row[3]))

    if run_nr is None:
        raise ValueError(f"Missing 'run_nr' in run log: {run_log_path}")

    if trigger_t0_us is None:
        raise ValueError(f"Missing 'trigger_t0_us' in run log: {run_log_path}")

    return (
        trigger_t0_us,
        run_nr,
        np.asarray(time_us, dtype=np.int64),
        np.asarray(press_bar, dtype=np.float64),
    )


# ==============================================================================
# VELOCITY PROFILE HELPERS
# ==============================================================================
def _read_velocity_csv(vel_csv: Path) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
]:
    """Read a velocity_final.csv file.

    Returns (win_x, win_y, time_s, vx_m_s, vy_m_s). time_s is corrected for the
    trigger/opening delay in the same way as the flow-rate time array, so the
    same START_STEP_MS/END_STEP_MS window used for flow also applies here.
    """
    data = np.genfromtxt(vel_csv, delimiter=",", names=True)
    if data.size == 0:
        raise RuntimeError(f"No data found in {vel_csv}")
    if getattr(data, "ndim", 0) == 0:
        data = np.array([data], dtype=data.dtype)

    win_x = np.asarray(data["win_x"], dtype=float).reshape(-1)
    win_y = np.asarray(data["win_y"], dtype=float).reshape(-1)
    time_s = np.asarray(data["time_s"], dtype=float).reshape(-1)
    vx = np.asarray(data["vx_m_s"], dtype=float).reshape(-1)
    vy = np.asarray(data["vy_m_s"], dtype=float).reshape(-1)

    finite = (
        np.isfinite(win_x) & np.isfinite(win_y) & np.isfinite(time_s)
        & np.isfinite(vx) & np.isfinite(vy)
    )
    win_x, win_y, time_s, vx, vy = (
        win_x[finite], win_y[finite], time_s[finite], vx[finite], vy[finite],
    )
    if time_s.size == 0:
        raise RuntimeError(f"No finite data found in {vel_csv}")

    # Same delay correction as applied to the flow-rate time array.
    time_s = time_s - (TRIGGER_DELAY_MS + OPENING_DELAY_MS) / 1000

    return win_x, win_y, time_s, vx, vy


def _step_window_stats(
    win_x: np.ndarray, win_y: np.ndarray, time_s: np.ndarray,
    vx: np.ndarray, vy: np.ndarray,
) -> tuple[
    dict[tuple[float, float], dict[str, float]],
    dict[tuple[float, float], tuple[np.ndarray, np.ndarray]],
]:
    """Median/std of vx, vy per window during the step interval.

    Returns (stats, raw): stats maps a (win_x, win_y) key to
    {"vx_med", "vx_std", "vy_med", "vy_std", "n"}; raw maps the same key to the
    underlying (vx_values, vy_values) samples so several runs can later be
    pooled together for an accumulated profile.
    """
    step_mask = (time_s >= START_STEP_MS /
                 1000) & (time_s < END_STEP_MS / 1000)
    if not np.any(step_mask):
        raise RuntimeError(
            f"No velocity samples found in the step interval "
            f"[{START_STEP_MS}, {END_STEP_MS}) ms"
        )

    wx_s, wy_s = win_x[step_mask], win_y[step_mask]
    vx_s, vy_s = vx[step_mask], vy[step_mask]

    uniq, inverse = np.unique(
        np.stack([wx_s, wy_s], axis=1), axis=0, return_inverse=True)
    inverse = np.asarray(inverse).reshape(-1)

    stats: dict[tuple[float, float], dict[str, float]] = {}
    raw: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]] = {}
    for i, (wx, wy) in enumerate(uniq):
        sel = inverse == i
        vx_w, vy_w = vx_s[sel], vy_s[sel]
        key = (float(wx), float(wy))
        stats[key] = {
            "vx_med": float(np.nanmedian(vx_w)),
            "vx_std": float(np.nanstd(vx_w)),
            "vy_med": float(np.nanmedian(vy_w)),
            "vy_std": float(np.nanstd(vy_w)),
            "n": int(vx_w.size),
        }
        raw[key] = (vx_w, vy_w)

    return stats, raw


def _format_coord(value: float) -> str:
    """Format a window coordinate for use in a CSV column name."""
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:g}".replace(".", "p").replace("-", "m")


def _window_column_prefix(win_x: float, win_y: float) -> str:
    return f"winx{_format_coord(win_x)}_winy{_format_coord(win_y)}"


def _window_position_mm_map(
    window_keys: list[tuple[float, float]],
) -> dict[tuple[float, float], tuple[float, float]]:
    """Map each window key to its centered X/Y position in millimetres."""
    return {
        key: (float(pos_x_mm), float(pos_y_mm))
        for key, pos_x_mm, pos_y_mm in zip(
            window_keys, win_pos_x_mm, win_pos_y_mm, strict=True
        )
    }


def _add_window_columns(
    row: dict,
    window_stats: dict[tuple[float, float], dict[str, float]],
    window_positions_mm: dict[tuple[float, float], tuple[float, float]],
) -> None:
    """Add per-window vx/vy median & std columns to a summary row dict, in place."""
    for (wx, wy), s in window_stats.items():
        prefix = _window_column_prefix(wx, wy)
        win_pos_x_mm, win_pos_y_mm = window_positions_mm[(wx, wy)]
        row[f"{prefix}_win_pos_x_mm"] = win_pos_x_mm
        row[f"{prefix}_win_pos_y_mm"] = win_pos_y_mm
        row[f"{prefix}_vx_med_m_s"] = s["vx_med"]
        row[f"{prefix}_vx_std_m_s"] = s["vx_std"]
        row[f"{prefix}_vy_med_m_s"] = s["vy_med"]
        row[f"{prefix}_vy_std_m_s"] = s["vy_std"]


def _center_window_keys(
    window_keys: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Return the middle four windows in a stable spatial order."""
    if len(window_keys) < 4:
        raise ValueError(
            f"Need at least 4 windows to compute center velocity, got {len(window_keys)}"
        )

    start = (len(window_keys) - 4) // 2
    return window_keys[start:start + 4]


def _center_velocity_stats(
    window_raw: dict[tuple[float, float], tuple[np.ndarray, np.ndarray]],
    center_keys: list[tuple[float, float]],
) -> dict[str, float]:
    """Calculate center velocity from the pooled middle four windows."""
    vx_all = np.concatenate([window_raw[key][0] for key in center_keys])
    vy_all = np.concatenate([window_raw[key][1] for key in center_keys])
    return {
        "vxc_avg_m_s": float(np.nanmean(vx_all)),
        "vxc_std_m_s": float(np.nanstd(vx_all)),
        "vyc_avg_m_s": float(np.nanmean(vy_all)),
        "vyc_std_m_s": float(np.nanstd(vy_all)),
    }


def _profile_positions(wx: np.ndarray, wy: np.ndarray,
                       scale_m_per_px: float) -> tuple[np.ndarray, str]:
    """Pick a 1D position axis for a velocity-profile plot, centre, and convert to m.

    If the windows form a line (one of win_x/win_y is constant), use the
    varying coordinate directly. Otherwise fall back to a plain window index.
    """

    # Centre the window coordinates around 0
    wx = wx - np.mean(wx)
    wy = wy - np.mean(wy)

    ux, uy = np.unique(wx), np.unique(wy)
    if uy.size > 1 and ux.size <= 1:
        return wy * scale_m_per_px * 1000, "Position (mm)"
    if ux.size > 1 and uy.size <= 1:
        return wx * scale_m_per_px * 1000, "Position (mm)"
    return np.arange(wx.size, dtype=float), "window index"


def _plot_velocity_profiles(
    labels: list[str],
    per_run_window_stats: list[dict[tuple[float, float], dict[str, float]]],
    accumulated_window_stats: dict[tuple[float, float], dict[str, float]],
    window_pos_x: np.ndarray,
    window_pos_y: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    """Plot per-run and accumulated velocity profiles (vx, vy vs window position).

    One subplot per run plus one for the accumulated profile, all sharing the
    same axis ranges so they can be compared directly.
    """
    panels = list(zip(labels, per_run_window_stats)) + \
        [("Accumulated (all runs)", accumulated_window_stats)]

    n_cols = 4
    n_rows = -(-len(panels) // n_cols)  # ceil division
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4.5 * n_cols, 6 * n_rows),
        sharex=True, sharey=True, squeeze=False,
    )
    axes_flat = axes.flatten()

    wx = window_pos_x
    wy = window_pos_y

    for ax, (label, window_stats) in zip(axes_flat, panels):
        keys = sorted(window_stats.keys())
        # wx = np.array([k[0] for k in keys])
        # wy = np.array([k[1] for k in keys])
        vx_med = np.array([window_stats[k]["vx_med"] for k in keys])
        vx_std = np.array([window_stats[k]["vx_std"] for k in keys])
        vy_med = np.array([window_stats[k]["vy_med"] for k in keys])
        vy_std = np.array([window_stats[k]["vy_std"] for k in keys])

        pos, pos_label = _profile_positions(wx, wy, SCALE_M_PER_PX)
        order = np.argsort(pos)

        ax.errorbar(
            vx_med[order], pos[order], xerr=vx_std[order],
            marker="o", ms=4, capsize=3, color="tab:blue", label="vx",
        )
        ax.errorbar(
            vy_med[order], pos[order], xerr=vy_std[order],
            marker="s", ms=4, capsize=3, color="tab:orange", label="vy",
        )
        ax.set_title(label)
        ax.set_xlabel("Velocity (m/s)")
        ax.grid(True, alpha=0.3)

    for ax in axes_flat[len(panels):]:
        ax.set_visible(False)

    axes_flat[0].set_ylabel(pos_label)
    axes_flat[-1].legend(loc="center")

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# ==============================================================================
# FLOW AND PRESSURE ANALYSIS
# ==============================================================================
# Load the run-log files once so each manifest row can be matched to its pressure trace.
run_logs_dir = _find_run_logs_dir(manifest_path, top_dir_path)
run_log_paths = sorted(run_logs_dir.rglob("*.csv"))
if not run_log_paths:
    raise RuntimeError(f"No run log CSV files found in {run_logs_dir}")

if len(run_log_paths) != len(flow_series):
    print(
        f"Warning: found {len(run_log_paths)} run logs in {run_logs_dir}, "
        f"but {len(flow_series)} flow runs in the manifest. Using pairwise order."
    )

# Calculate the flow statistics and the step-pressure statistics for each run.
print(
    f"\nFlow rate analysis for batch run: {batch_run_id} with pressure {pressure} bar and valve current {current} mA")

COL_W = 22  # width of each "avg ± std" column


def fmt_row(label, base_med, base_std, step_med, step_std, diff_med, diff_std, delay_avg, delay_std, step_press_med, step_press_std):
    base_cell = f"{base_med:.4f} ± {base_std:.4f}"
    step_cell = f"{step_med:.4f} ± {step_std:.4f}"
    diff_cell = f"{diff_med:.4f} ± {diff_std:.4f}"
    delay_cell = f"{delay_avg:.1f} ± {delay_std:.1f}"
    pressure_cell = f"{step_press_med:.4f} ± {step_press_std:.4f}"
    return f"{label:<7}" + "".join(f"{c:^{COL_W}}" for c in [base_cell, step_cell, diff_cell, delay_cell, pressure_cell])


header = f"{'Run':<7}" + "".join(
    f"{h:^{COL_W}}" for h in
    ["Base med ± std (L/s)", "Step med ± std (L/s)",
     "Diff med ± std (L/s)", "Delay med ± std (ms)", "Pressure ± std (bar)"]
)

fancy_labels = []
summary_rows = []
delays = []
step_pressure_means: list[float] = []
step_pressure_stds: list[float] = []
flow_rows: list[tuple[str, float, float, float,
                      float, float, float, float, float]] = []
per_run_window_stats: list[dict[tuple[float, float], dict[str, float]]] = []
per_run_window_raw: list[dict[tuple[float, float],
                              tuple[np.ndarray, np.ndarray]]] = []
for index, ((label, time_s, flow_lps), run_log_path) in enumerate(
    zip(flow_series, run_log_paths[: len(flow_series)])
):
    # Calculate average and std dev over the base and step intervals
    base_mask = (time_s >= START_BASE_MS /
                 1000) & (time_s < END_BASE_MS / 1000)
    step_mask = (time_s >= START_STEP_MS /
                 1000) & (time_s < END_STEP_MS / 1000)

    base_med = np.nanmedian(flow_lps[base_mask])
    base_std = np.nanstd(flow_lps[base_mask])

    step_med = np.nanmedian(flow_lps[step_mask])
    step_std = np.nanstd(flow_lps[step_mask])

    # Compute the difference and its std dev (assuming independence)
    diff_med = step_med - base_med
    diff_std = np.sqrt(base_std**2 + step_std**2)

    # Calculate the flow delay (aka the point in time where the flow rate first starts to rise)
    delay = np.nanmin(
        time_s[flow_lps > (STEP_START_THRESHOLD_FACTOR*base_med)]) * 1000
    # delay = np.nanmin(time_s[flow_lps > (1.01*base_med)]) * 1000
    flow_rows.append((label, base_med, base_std, step_med,
                     step_std, diff_med, diff_std, delay, 0.0))

    # Read the matching run log and compute the mean pressure during the step interval.
    trigger_t0_us, run_nr, time_us, press_bar = _read_run_log(run_log_path)
    time_ms = (time_us - trigger_t0_us) / 1000.0

    # Align the time array to account for the opening delay and the flow delay
    # (sort of: the pressure when the valve was opened)
    aligned_time_ms = time_ms + OPENING_DELAY_MS + delay
    step_pressure_mask = (
        (aligned_time_ms >= START_STEP_MS) & (aligned_time_ms < END_STEP_MS)
    )
    step_pressure_values = press_bar[step_pressure_mask]
    if step_pressure_values.size == 0:
        raise RuntimeError(
            f"No pressure samples found in interval [{START_STEP_MS}, {END_STEP_MS}) ms for {run_log_path}"
        )
    step_pressure_med_bar = float(np.nanmedian(step_pressure_values))
    step_pressure_std_bar = float(np.nanstd(step_pressure_values))

    step_pressure_means.append(step_pressure_med_bar)
    step_pressure_stds.append(step_pressure_std_bar)

    delays.append(delay)

    summary_rows.append({
        "row_type": "run",
        "label": label,
        "pressure_bar": pressure,
        "current_mA": current,
        "base_start_ms": START_BASE_MS,
        "base_end_ms": END_BASE_MS,
        "base_med_L_s": base_med,
        "base_std_L_s": base_std,
        "step_start_ms": START_STEP_MS,
        "step_end_ms": END_STEP_MS,
        "step_med_L_s": step_med,
        "step_std_L_s": step_std,
        "diff_med_L_s": diff_med,
        "diff_std_L_s": diff_std,
        "step_start_threshold_factor": STEP_START_THRESHOLD_FACTOR,
        "delay_avg_ms": delay,
        "delay_std_ms": 0.0,  # single run, so no std dev
        "step_pressure_bar": step_pressure_med_bar,
        "step_pressure_std_bar": step_pressure_std_bar,
    })

    # ALso generate a label for the plot that includes the diff +- std
    fancy_labels.append(f"{label} ({diff_med:.2f} ± {diff_std:.2f} L/s)")

    # Velocity profile during the step interval, from velocity_final.csv
    win_x_v, win_y_v, time_s_v, vx_v, vy_v = _read_velocity_csv(
        velocity_paths[index])
    window_stats, window_raw = _step_window_stats(
        win_x_v, win_y_v, time_s_v, vx_v, vy_v)
    per_run_window_stats.append(window_stats)
    per_run_window_raw.append(window_raw)
    center_keys = _center_window_keys(sorted(window_stats.keys()))
    center_stats = _center_velocity_stats(window_raw, center_keys)
    run_window_positions_mm = _window_position_mm_map(
        sorted(window_stats.keys()))
    _add_window_columns(
        summary_rows[-1], window_stats, run_window_positions_mm)
    summary_rows[-1].update(center_stats)


# Finally add a row for the accumulated average and std dev across all runs
all_base_flows = np.concatenate([
    flow_lps[(time_s >= START_BASE_MS / 1000) & (time_s < END_BASE_MS / 1000)]
    for _, time_s, flow_lps in flow_series
])
all_step_flows = np.concatenate([
    flow_lps[(time_s >= START_STEP_MS / 1000) & (time_s < END_STEP_MS / 1000)]
    for _, time_s, flow_lps in flow_series
])

all_base_med = np.nanmedian(all_base_flows)
all_base_std = np.nanstd(all_base_flows)
all_step_med = np.nanmedian(all_step_flows)
all_step_std = np.nanstd(all_step_flows)
all_diff_med = all_step_med - all_base_med
all_diff_std = np.sqrt(all_base_std**2 + all_step_std**2)
# Average delay is the average of the individual delays, not the delay of the average flow rate
all_delay_avg = np.nanmean(delays)
all_delay_std = np.nanstd(delays)


all_step_pressure_med = float(np.nanmean(step_pressure_means))
all_step_pressure_std = float(np.nanstd(step_pressure_means))

# Accumulated velocity profile: pool the raw step-interval samples for each
# window across all runs, then take the median/std of the pooled samples
# (mirrors how all_step_med/all_step_std are computed for flow, above).
all_window_keys = sorted(
    set().union(*(raw.keys() for raw in per_run_window_raw)))
all_window_positions_mm = _window_position_mm_map(all_window_keys)
accumulated_window_stats: dict[tuple[float, float], dict[str, float]] = {}
for key in all_window_keys:
    vx_all = np.concatenate(
        [raw[key][0] for raw in per_run_window_raw if key in raw])
    vy_all = np.concatenate(
        [raw[key][1] for raw in per_run_window_raw if key in raw])
    accumulated_window_stats[key] = {
        "vx_med": float(np.nanmedian(vx_all)),
        "vx_std": float(np.nanstd(vx_all)),
        "vy_med": float(np.nanmedian(vy_all)),
        "vy_std": float(np.nanstd(vy_all)),
        "n": int(vx_all.size),
    }

all_center_keys = _center_window_keys(all_window_keys)
all_center_stats = _center_velocity_stats(
    {key: (np.concatenate([raw[key][0] for raw in per_run_window_raw if key in raw]),
           np.concatenate([raw[key][1] for raw in per_run_window_raw if key in raw]))
     for key in all_center_keys},
    all_center_keys,
)

# Print the combined table once all per-run and batch-level statistics are ready.
print(header)
print("-" * len(header))
for (label, base_med, base_std, step_med, step_std, diff_med, diff_std, delay, delay_std), step_pressure_med, step_pressure_std in zip(flow_rows, step_pressure_means, step_pressure_stds):
    print(fmt_row(label, base_med, base_std, step_med, step_std,
          diff_med, diff_std, delay, delay_std, step_pressure_med, step_pressure_std))
print("-" * len(header))
print(fmt_row("ALL", all_base_med, all_base_std,
      all_step_med, all_step_std, all_diff_med, all_diff_std, all_delay_avg, all_delay_std, all_step_pressure_med, all_step_pressure_std))

# Write a summary CSV that includes both flow and step-pressure statistics.
summary_rows.append({
    "row_type": "all",
    "label": "ALL",
    "pressure_bar": pressure,
    "current_mA": current,
    "base_start_ms": START_BASE_MS,
    "base_end_ms": END_BASE_MS,
    "base_med_L_s": all_base_med,
    "base_std_L_s": all_base_std,
    "step_start_ms": START_STEP_MS,
    "step_end_ms": END_STEP_MS,
    "step_med_L_s": all_step_med,
    "step_std_L_s": all_step_std,
    "diff_med_L_s": all_diff_med,
    "diff_std_L_s": all_diff_std,
    "delay_avg_ms": all_delay_avg,
    "delay_std_ms": all_delay_std,
    "step_start_threshold_factor": STEP_START_THRESHOLD_FACTOR,
    "step_pressure_bar": all_step_pressure_med,
    "step_pressure_std_bar": all_step_pressure_std,
})
summary_rows[-1].update(all_center_stats)
_add_window_columns(
    summary_rows[-1], accumulated_window_stats, all_window_positions_mm)

# Per-window velocity columns: 5 columns (profile position plus vx/vy med/std)
# for every window position found across all runs, in a consistent order.
window_fieldnames: list[str] = []
for wx, wy in all_window_keys:
    prefix = _window_column_prefix(wx, wy)
    window_fieldnames.extend([
        f"{prefix}_win_pos_x_mm",
        f"{prefix}_win_pos_y_mm",
        f"{prefix}_vx_med_m_s",
        f"{prefix}_vx_std_m_s",
        f"{prefix}_vy_med_m_s",
        f"{prefix}_vy_std_m_s",
    ])

expanded_manifest_path = output_dir / f"{batch_run_id}_flow_rate_summary.csv"
with expanded_manifest_path.open("w", newline="", encoding="utf-8") as fp:
    writer = csv.DictWriter(fp, fieldnames=[
        "row_type",
        "label",
        "pressure_bar",
        "current_mA",
        "base_start_ms",
        "base_end_ms",
        "base_med_L_s",
        "base_std_L_s",
        "step_start_ms",
        "step_end_ms",
        "step_med_L_s",
        "step_std_L_s",
        "diff_med_L_s",
        "diff_std_L_s",
        "delay_avg_ms",
        "delay_std_ms",
        "step_start_threshold_factor",
        "step_pressure_bar",
        "step_pressure_std_bar",
        "vxc_avg_m_s",
        "vxc_std_m_s",
        "vyc_avg_m_s",
        "vyc_std_m_s",
    ] + window_fieldnames)
    writer.writeheader()
    writer.writerows(summary_rows)

# Before plotting, rename the labels in flow_series to include the fancy labels with diff ± std
flow_series = [(fancy_label, time_s, flow_lps)
               for fancy_label, (_, time_s, flow_lps) in zip(fancy_labels, flow_series)
               ]
plot_title = f"{current} mA step flow at {all_step_pressure_med:.2f} ± {all_step_pressure_std:.2f} bar"

# Regenerate plot
series_plot_path = output_dir / f"{batch_run_id}_flow_rate_comparison.pdf"
viz.plot_flow_rate_series(
    flow_series,
    title=plot_title,
    output_path=series_plot_path,
    interval1=(START_BASE_MS/1000, END_BASE_MS/1000),
    interval2=(START_STEP_MS/1000, END_STEP_MS/1000),
    interval1_label=f"Closed med.: ({all_base_med:.2f} ± {all_base_std:.2f} L/s)",
    interval2_label=f"Open med.: ({all_step_med:.2f} ± {all_step_std:.2f} L/s)",
    vertical_lines=[all_delay_avg],
    subtitle=f"Compensated for nebuliser: {all_diff_med:.2f} ± {all_diff_std:.2f} L/s; avg. delay: {all_delay_avg:.1f} ± {all_delay_std:.1f} ms",
)

# Velocity profile plot: one subplot per run plus the accumulated profile,
# all sharing the same axis ranges.
velocity_plot_path = output_dir / f"{batch_run_id}_velocity_profile.pdf"
_plot_velocity_profiles(
    labels=labels,
    per_run_window_stats=per_run_window_stats,
    accumulated_window_stats=accumulated_window_stats,
    window_pos_x=WIN_POS_X,
    window_pos_y=WIN_POS_Y,
    output_path=velocity_plot_path,
    title=f"Velocity profile during step interval ({START_STEP_MS}-{END_STEP_MS} ms) — {batch_run_id}",
)
print(f"Saved velocity profile plot to {velocity_plot_path}")
