from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tcm_utils.file_dialogs import ask_open_file
from tcm_utils.plot_style import use_tcm_poster_style, append_unit_to_last_ticklabel
from tcm_utils.cvd_check import set_cvd_friendly_colors, get_color
from tcm_control import read_run_log


TRIGGER_DELAY_MS = -10
OPENING_DELAY_MS = 50
FLOW_TRAVEL_DELAY_MS = 6
BASELINE_START_MS = -50
BASELINE_END_MS = 0
BIN_WIDTH_MS = 5
LOWER_PERCENTILE = 5
UPPER_PERCENTILE = 95

# Temporary working dataset paths.
WORKING_RUNS = [
    (
        Path("/Volumes/Data/PIV/260820_piv/final/260910_173132_my_cough/run_logs/log1_260910_173225.csv"),
        Path("/Volumes/Data/PIV/260820_piv/final/260910_173132_my_cough/260914_132007_p-001_restitched_flow_rate.csv"),
    ),
    (
        Path("/Volumes/Data/PIV/260820_piv/final/260910_173132_my_cough/run_logs/log2_260910_173350.csv"),
        Path("/Volumes/Data/PIV/260820_piv/final/260910_173132_my_cough/260914_132007_p-002_restitched_flow_rate.csv"),
    ),
    (
        Path("/Volumes/Data/PIV/260820_piv/final/260910_173132_my_cough/run_logs/log3_260910_173516.csv"),
        Path("/Volumes/Data/PIV/260820_piv/final/260910_173132_my_cough/260914_132007_p-003_restitched_flow_rate.csv"),
    ),
]


def read_flow_rate_csv(flow_rate_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(flow_rate_path, delimiter=",", names=True)
    if data.size == 0:
        raise RuntimeError(f"No data found in {flow_rate_path}")
    if getattr(data, "ndim", 0) == 0:
        data = np.array([data], dtype=data.dtype)

    if "time_s" not in data.dtype.names or "flow_rate_L_s" not in data.dtype.names:
        raise ValueError(
            "flow_rate.csv must contain time_s and flow_rate_L_s columns"
        )

    return (
        np.asarray(data["time_s"], dtype=np.float64),
        np.asarray(data["flow_rate_L_s"], dtype=np.float64),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot the measured data from a human-cough run log."
    )
    parser.add_argument(
        "run_log",
        nargs="?",
        type=Path,
        help="Path to the run log CSV. If omitted, a file picker opens.",
    )
    parser.add_argument(
        "flow_rate_csv",
        nargs="?",
        type=Path,
        help="Path to the stitched PIV flow_rate.csv. If omitted, a file picker opens.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help="Optional experiment directory used to save the PDF in processed/",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive plot window.",
    )
    args = parser.parse_args()

    experiment_dir = (
        args.experiment_dir.expanduser().resolve()
        if args.experiment_dir is not None
        else WORKING_RUNS[0][0].parent.parent
    )

    use_tcm_poster_style()
    set_cvd_friendly_colors()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_ylabel("Flow rate (L/s)")

    all_measured_time_ms: list[np.ndarray] = []
    all_measured_flow_lps: list[np.ndarray] = []

    for index, (run_log_path, flow_rate_path) in enumerate(WORKING_RUNS):
        (
            trigger_t0_us,
            run_nr,
            time_us,
            sol_valve_action,
            req_flow_lps,
            prop_valve_ma,
            press_bar,
        ) = read_run_log(run_log_path)

        flow_time_s, flow_rate_lps = read_flow_rate_csv(flow_rate_path)

        flow_time_ms = (
            flow_time_s * 1000.0
            - TRIGGER_DELAY_MS
            - FLOW_TRAVEL_DELAY_MS
            - OPENING_DELAY_MS
        )

        baseline_mask = (flow_time_ms >= BASELINE_START_MS) & (
            flow_time_ms <= BASELINE_END_MS)
        if not np.any(baseline_mask):
            raise RuntimeError(
                f"No baseline samples found in {flow_rate_path}"
            )

        flow_rate_lps = flow_rate_lps - np.median(
            flow_rate_lps[baseline_mask]
        )
        all_measured_time_ms.append(flow_time_ms)
        all_measured_flow_lps.append(flow_rate_lps)
        time_ms = (
            (time_us - trigger_t0_us) / 1000.0 - OPENING_DELAY_MS
        )

        # # The three stitched PIV measurements: same colour, transparent.
        # ax.plot(
        #     flow_time_ms / 1000.0,
        #     flow_rate_lps,
        #     linewidth=2,
        #     color=get_color(0),
        #     alpha=0.35,
        #     label="Flow at cough machine exit,\nmeasured 3× using PIV" if index == 0 else "_nolegend_",
        # )

    # Pool all measured samples into 1 ms bins and calculate median and IQR.
    pooled_time_ms = np.concatenate(all_measured_time_ms)
    pooled_flow_lps = np.concatenate(all_measured_flow_lps)

    bin_edges_ms = np.arange(-50.0, 651.0, BIN_WIDTH_MS)
    bin_centres_ms = bin_edges_ms[:-1] + BIN_WIDTH_MS / 2.0

    bin_index = np.digitize(pooled_time_ms, bin_edges_ms) - 1
    valid_samples = (bin_index >= 0) & (bin_index < len(bin_centres_ms))

    median_flow_lps = np.full(len(bin_centres_ms), np.nan)
    lower_percentile_lps = np.full(len(bin_centres_ms), np.nan)
    upper_percentile_lps = np.full(len(bin_centres_ms), np.nan)

    for bin_number in range(len(bin_centres_ms)):
        values = pooled_flow_lps[
            valid_samples & (bin_index == bin_number)
        ]

        if values.size == 0:
            continue

        lower_percentile_lps[bin_number] = np.percentile(
            values, LOWER_PERCENTILE)
        median_flow_lps[bin_number] = np.median(values)
        upper_percentile_lps[bin_number] = np.percentile(
            values, UPPER_PERCENTILE)

    has_data = np.isfinite(median_flow_lps)

    # IQR: shaded 25th–75th percentile band.
    ax.fill_between(
        bin_centres_ms[has_data] / 1000.0,
        lower_percentile_lps[has_data],
        upper_percentile_lps[has_data],
        color=get_color(0),
        alpha=0.5,
        linewidth=0,
        label="_nolegend_",
    )

    # Median measured flow.
    ax.plot(
        bin_centres_ms[has_data] / 1000.0,
        median_flow_lps[has_data],
        color=get_color(0),
        linewidth=2,
        label=f"Flow at cough machine exit:\nmedian (with 5-95% interval)\nof 3 repeats in {BIN_WIDTH_MS} ms bins,\nshifted to line up start of flow",
    )

    # Plot the requested-flow curve for one case
    ax.plot(
        time_ms[req_flow_lps >= 0] / 1000.0,
        req_flow_lps[req_flow_lps >= 0],
        color=get_color(2),
        linewidth=2,
        alpha=1,
        label=(
            "Input: Gupta et al. ('09) model\nfor a 71 kg, 1.94 m male"
        ),
    )

    # Add a horizontal line at y=0
    ax.axhline(0, color="black", linewidth=2, alpha=0.3, linestyle="-")

    ax.set_xlim(-0.05, 0.55)
    # ax.set_xlim(0, 0.02)
    ax.set_title(f"Flow rate comparison")
    ax.grid(True, linestyle=":", alpha=0.35)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        [handles[1], handles[0]],
        [labels[1], labels[0]],
        loc="upper right",
        fontsize=12,
    )
    append_unit_to_last_ticklabel(ax, axis="x", unit="s")
    plot_path = experiment_dir / "flow_rate_comparison.pdf"
    fig.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

    if args.no_show:
        plt.close(fig)
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
