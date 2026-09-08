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
START_STEP_MS = 65
END_STEP_MS = 95

# Set the factor by which the flow must rise above the base average to be considered a step response
RISE_FACTOR = 1.0

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


def fmt_row(label, base_avg, base_std, step_avg, step_std, diff_avg, diff_std, delay_avg, delay_std, step_press_avg, step_press_std):
    base_cell = f"{base_avg:.4f} ± {base_std:.4f}"
    step_cell = f"{step_avg:.4f} ± {step_std:.4f}"
    diff_cell = f"{diff_avg:.4f} ± {diff_std:.4f}"
    delay_cell = f"{delay_avg:.1f} ± {delay_std:.1f}"
    pressure_cell = f"{step_press_avg:.4f} ± {step_press_std:.4f}"
    return f"{label:<7}" + "".join(f"{c:^{COL_W}}" for c in [base_cell, step_cell, diff_cell, delay_cell, pressure_cell])


header = f"{'Run':<7}" + "".join(
    f"{h:^{COL_W}}" for h in
    ["Base avg ± std (L/s)", "Step avg ± std (L/s)",
     "Diff avg ± std (L/s)", "Delay avg ± std (ms)", "Pressure ± std (bar)"]
)

fancy_labels = []
summary_rows = []
delays = []
step_pressure_means: list[float] = []
step_pressure_stds: list[float] = []
flow_rows: list[tuple[str, float, float, float,
                      float, float, float, float, float]] = []
for index, ((label, time_s, flow_lps), run_log_path) in enumerate(
    zip(flow_series, run_log_paths[: len(flow_series)])
):
    # Calculate average and std dev over the base and step intervals
    base_mask = (time_s >= START_BASE_MS /
                 1000) & (time_s < END_BASE_MS / 1000)
    step_mask = (time_s >= START_STEP_MS /
                 1000) & (time_s < END_STEP_MS / 1000)

    base_avg = np.nanmean(flow_lps[base_mask])
    base_std = np.nanstd(flow_lps[base_mask])

    step_avg = np.nanmean(flow_lps[step_mask])
    step_std = np.nanstd(flow_lps[step_mask])

    # Compute the difference and its std dev (assuming independence)
    diff_avg = step_avg - base_avg
    diff_std = np.sqrt(base_std**2 + step_std**2)

    # Calculate the flow delay (aka the point in time where the flow rate first starts to rise)
    delay = np.nanmin(time_s[flow_lps > (RISE_FACTOR*base_avg)]) * 1000
    # delay = np.nanmin(time_s[flow_lps > (1.01*base_avg)]) * 1000
    flow_rows.append((label, base_avg, base_std, step_avg,
                     step_std, diff_avg, diff_std, delay, 0.0))

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
    step_pressure_avg_bar = float(np.nanmean(step_pressure_values))
    step_pressure_std_bar = float(np.nanstd(step_pressure_values))

    step_pressure_means.append(step_pressure_avg_bar)
    step_pressure_stds.append(step_pressure_std_bar)

    delays.append(delay)

    summary_rows.append({
        "row_type": "run",
        "label": label,
        "pressure_bar": pressure,
        "current_mA": current,
        "base_start_ms": START_BASE_MS,
        "base_end_ms": END_BASE_MS,
        "base_avg_L_s": base_avg,
        "base_std_L_s": base_std,
        "step_start_ms": START_STEP_MS,
        "step_end_ms": END_STEP_MS,
        "step_avg_L_s": step_avg,
        "step_std_L_s": step_std,
        "diff_avg_L_s": diff_avg,
        "diff_std_L_s": diff_std,
        "delay_avg_ms": delay,
        "delay_std_ms": 0.0,  # single run, so no std dev
        "step_pressure_bar": step_pressure_avg_bar,
        "step_pressure_std_bar": step_pressure_std_bar,
    })

    # ALso generate a label for the plot that includes the diff +- std
    fancy_labels.append(f"{label} ({diff_avg:.2f} ± {diff_std:.2f} L/s)")


# Finally add a row for the accumulated average and std dev across all runs
all_base_flows = np.concatenate([
    flow_lps[(time_s >= START_BASE_MS / 1000) & (time_s < END_BASE_MS / 1000)]
    for _, time_s, flow_lps in flow_series
])
all_step_flows = np.concatenate([
    flow_lps[(time_s >= START_STEP_MS / 1000) & (time_s < END_STEP_MS / 1000)]
    for _, time_s, flow_lps in flow_series
])

all_base_avg = np.nanmean(all_base_flows)
all_base_std = np.nanstd(all_base_flows)
all_step_avg = np.nanmean(all_step_flows)
all_step_std = np.nanstd(all_step_flows)
all_diff_avg = all_step_avg - all_base_avg
all_diff_std = np.sqrt(all_base_std**2 + all_step_std**2)
# Average delay is the average of the individual delays, not the delay of the average flow rate
all_delay_avg = np.nanmean(delays)
all_delay_std = np.nanstd(delays)


all_step_pressure_avg = float(np.nanmean(step_pressure_means))
all_step_pressure_std = float(np.nanstd(step_pressure_means))

# Print the combined table once all per-run and batch-level statistics are ready.
print(header)
print("-" * len(header))
for (label, base_avg, base_std, step_avg, step_std, diff_avg, diff_std, delay, delay_std), step_pressure_avg, step_pressure_std in zip(flow_rows, step_pressure_means, step_pressure_stds):
    print(fmt_row(label, base_avg, base_std, step_avg, step_std,
          diff_avg, diff_std, delay, delay_std, step_pressure_avg, step_pressure_std))
print("-" * len(header))
print(fmt_row("ALL", all_base_avg, all_base_std,
      all_step_avg, all_step_std, all_diff_avg, all_diff_std, all_delay_avg, all_delay_std, all_step_pressure_avg, all_step_pressure_std))

# Write a summary CSV that includes both flow and step-pressure statistics.
summary_rows.append({
    "row_type": "all",
    "label": "ALL",
    "pressure_bar": pressure,
    "current_mA": current,
    "base_start_ms": START_BASE_MS,
    "base_end_ms": END_BASE_MS,
    "base_avg_L_s": all_base_avg,
    "base_std_L_s": all_base_std,
    "step_start_ms": START_STEP_MS,
    "step_end_ms": END_STEP_MS,
    "step_avg_L_s": all_step_avg,
    "step_std_L_s": all_step_std,
    "diff_avg_L_s": all_diff_avg,
    "diff_std_L_s": all_diff_std,
    "delay_avg_ms": all_delay_avg,
    "delay_std_ms": all_delay_std,
    "step_pressure_bar": all_step_pressure_avg,
    "step_pressure_std_bar": all_step_pressure_std,
})

expanded_manifest_path = output_dir / f"{batch_run_id}_flow_rate_summary.csv"
with expanded_manifest_path.open("w", newline="", encoding="utf-8") as fp:
    writer = csv.DictWriter(fp, fieldnames=[
        "row_type",
        "label",
        "pressure_bar",
        "current_mA",
        "base_start_ms",
        "base_end_ms",
        "base_avg_L_s",
        "base_std_L_s",
        "step_start_ms",
        "step_end_ms",
        "step_avg_L_s",
        "step_std_L_s",
        "diff_avg_L_s",
        "diff_std_L_s",
        "delay_avg_ms",
        "delay_std_ms",
        "step_pressure_bar",
        "step_pressure_std_bar",
    ])
    writer.writeheader()
    writer.writerows(summary_rows)

# Before plotting, rename the labels in flow_series to include the fancy labels with diff ± std
flow_series = [(fancy_label, time_s, flow_lps)
               for fancy_label, (_, time_s, flow_lps) in zip(fancy_labels, flow_series)
               ]
plot_title = f"{current} mA step flow at {all_step_pressure_avg:.2f} ± {all_step_pressure_std:.2f} bar"

# Regenerate plot
series_plot_path = output_dir / f"{batch_run_id}_flow_rate_comparison.pdf"
viz.plot_flow_rate_series(
    flow_series,
    title=plot_title,
    output_path=series_plot_path,
    interval1=(START_BASE_MS/1000, END_BASE_MS/1000),
    interval2=(START_STEP_MS/1000, END_STEP_MS/1000),
    interval1_label=f"Closed avg: ({all_base_avg:.2f} ± {all_base_std:.2f} L/s)",
    interval2_label=f"Open avg: ({all_step_avg:.2f} ± {all_step_std:.2f} L/s)",
    vertical_lines=[all_delay_avg],
    subtitle=f"Compensated for nebuliser: {all_diff_avg:.2f} ± {all_diff_std:.2f} L/s; delay: {all_delay_avg:.1f} ± {all_delay_std:.1f} ms",
)
