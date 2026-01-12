"""Lightweight plotting utilities for tcm-piv.

These helpers keep plotting logic out of the main run loop so that
plots can be generated conditionally based on config flags.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from tcm_piv.preprocessing import split_n_shift


def gupta_model(gender: str, mass: float, height: float) -> tuple[np.ndarray, np.ndarray]:
    """Return a simple cough flow-rate model inspired by Gupta et al. (2009).

    This lightweight implementation avoids external dependencies. It creates a
    smooth, single-peaked waveform whose amplitude scales modestly with mass
    and height.
    """

    peak_l_s = 11.0 if gender.lower().startswith("m") else 8.0
    scale = (mass / 70.0) ** 0.3 * (height / 1.75) ** 0.3
    peak_l_s *= scale

    t = np.linspace(0.0, 0.15, 150)
    t0 = 0.05
    sigma = 0.02
    flow = peak_l_s * np.exp(-0.5 * ((t - t0) / sigma) ** 2)
    return t, flow


def plot_filter_ranges(
    ranges: list[tuple[float, float]], *, output_path: Path | None = None
) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.axhline(0.0, color="0.7", linewidth=1.0)
    ax.axvline(0.0, color="0.7", linewidth=1.0)

    for idx, (vx, vy) in enumerate(ranges, start=1):
        rect = Rectangle((-vx, -vy), 2 * vx, 2 * vy,
                         fill=False, linewidth=1.5, label=f"Pass {idx}")
        ax.add_patch(rect)

    if ranges:
        vx_max = max(abs(vx) for vx, _ in ranges)
        vy_max = max(abs(vy) for _, vy in ranges)
        pad_x = vx_max * 0.1 if vx_max else 1.0
        pad_y = vy_max * 0.1 if vy_max else 1.0
        ax.set_xlim(-(vx_max + pad_x), vx_max + pad_x)
        ax.set_ylim(-(vy_max + pad_y), vy_max + pad_y)

    ax.set_xlabel("v_x limit (m/s)")
    ax.set_ylabel("v_y limit (m/s)")
    ax.set_title("Global filter ranges")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.set_aspect("equal")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)

    return fig, ax


def plot_window_layout(
    image: np.ndarray,
    n_wins: tuple[int, int],
    *,
    overlap: float = 0.0,
    shifts: np.ndarray | None = None,
    shift_mode: str = "before",
    title: str | None = None,
    output_path: Path | None = None,
) -> tuple[Figure, Axes]:
    """Plot window rectangles using the exact logic from `split_n_shift`.

    Notes:
    - If `shifts` is per-pair (n_pairs, n_y, n_x, 2), the mean shift over time is used.
    - This is meant for debug/visualisation only; it does not affect computation.
    """

    shift_arg: tuple[int, int] | np.ndarray
    if shifts is None:
        shift_arg = (0, 0)
    else:
        shifts = np.asarray(shifts)
        if shifts.ndim == 4:
            shift_arg = np.nanmean(shifts, axis=0)
        else:
            shift_arg = shifts

    # `split_n_shift(plot=True)` creates its own figure/axes.
    split_n_shift(
        np.asarray(image),
        n_wins,
        overlap=float(overlap),
        shift=shift_arg,
        shift_mode=shift_mode,
        plot=True,
    )
    fig = plt.gcf()
    ax = plt.gca()
    if title:
        ax.set_title(title)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)

    return fig, ax


def plot_flow_rate(
    time_s: np.ndarray,
    flow_ls: np.ndarray,
    model: tuple[np.ndarray, np.ndarray] | None = None,
    *,
    output_path: Path | None = None,
) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(time_s * 1000.0, flow_ls, label="Measured", color="steelblue")

    if model is not None:
        t_model, q_model = model
        ax.plot(t_model * 1000.0, q_model, label="Model", color="orange")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Flow rate (L/s)")
    ax.set_title("Flow rate vs time")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(loc="upper right")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)

    return fig, ax
