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
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle, Rectangle, Wedge

from tcm_piv.preprocessing import split_n_shift


def plot_filter_ranges(
    ranges: list[tuple[float, float]],
    *,
    mode: str = "semicircle_rect",
    flow_direction: str = "x",
    output_path: Path | None = None,
) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.axhline(0.0, color="0.7", linewidth=1.0)
    ax.axvline(0.0, color="0.7", linewidth=1.0)

    mode_norm = str(mode).strip().lower()
    flow_dir = str(flow_direction).strip().lower()
    if flow_dir not in {"x", "y"}:
        raise ValueError("flow_direction must be 'x' or 'y'")

    for idx, (vx, vy) in enumerate(ranges, start=1):
        if mode_norm == "semicircle_rect":
            # Matches `filter_outliers('semicircle_rect', ...)`:
            # left half: circle of radius a (= vy) for vx < 0
            # right half: rectangle with vx in [0, b] (= vx) and vy in [-a, a]
            if flow_dir == "x":
                # streamwise = vx (x-axis), cross-stream = vy (y-axis)
                a = float(vy)
                b = float(vx)
                ax.add_patch(
                    Wedge(
                        (0.0, 0.0),
                        r=a,
                        theta1=90.0,
                        theta2=270.0,
                        fill=False,
                        linewidth=1.5,
                        label=f"Pass {idx}",
                    )
                )
                ax.add_patch(
                    Rectangle(
                        (0.0, -a),
                        b,
                        2 * a,
                        fill=False,
                        linewidth=1.5,
                    )
                )
            else:
                # streamwise = vy (y-axis), cross-stream = vx (x-axis)
                a = float(vx)
                b = float(vy)
                ax.add_patch(
                    Wedge(
                        (0.0, 0.0),
                        r=a,
                        theta1=180.0,
                        theta2=360.0,
                        fill=False,
                        linewidth=1.5,
                        label=f"Pass {idx}",
                    )
                )
                ax.add_patch(
                    Rectangle(
                        (-a, 0.0),
                        2 * a,
                        b,
                        fill=False,
                        linewidth=1.5,
                    )
                )
        elif mode_norm == "circle":
            r = float(max(abs(vx), abs(vy)))
            ax.add_patch(
                Circle(
                    (0.0, 0.0),
                    radius=r,
                    fill=False,
                    linewidth=1.5,
                    label=f"Pass {idx}",
                )
            )
        else:
            rect = Rectangle(
                (-vx, -vy),
                2 * vx,
                2 * vy,
                fill=False,
                linewidth=1.5,
                label=f"Pass {idx}",
            )
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
    ax.set_title(f"Global filter ranges ({mode_norm}, flow={flow_dir})")
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


def export_velocity_profiles_pdf(
    *,
    vel_final: np.ndarray,
    image: np.ndarray,
    n_wins: tuple[int, int],
    overlap: float,
    ds_fac: int,
    scale_m_per_px: float,
    flow_direction: str,
    time_s: np.ndarray,
    output_path: Path,
) -> None:
    """Export a multi-page PDF of per-pair velocity profiles.

    For each image pair (frame index), makes a plot where:
    - y-axis: signed distance from image centre based on window x-centres
    - x-axis: velocity components (m/s)

    Two curves are drawn:
    - streamwise component
    - cross-stream component
    """

    vel_final = np.asarray(vel_final)
    if vel_final.ndim != 4 or vel_final.shape[-1] != 2:
        raise ValueError("vel_final must have shape (n_pairs, n_y, n_x, 2)")

    n_pairs, n_y, n_x, _ = vel_final.shape
    if (n_y, n_x) != (int(n_wins[0]), int(n_wins[1])):
        # Keep it strict so we don't silently plot the wrong geometry.
        raise ValueError(
            f"n_wins {n_wins} does not match vel_final window grid {(n_y, n_x)}"
        )

    flow_dir = str(flow_direction).strip().lower()
    if flow_dir == "x":
        stream_idx, cross_idx = 1, 0  # vx, vy
        stream_label, cross_label = "streamwise (v_x)", "cross-stream (v_y)"
    elif flow_dir == "y":
        stream_idx, cross_idx = 0, 1  # vy, vx
        stream_label, cross_label = "streamwise (v_y)", "cross-stream (v_x)"
    else:
        raise ValueError("flow_direction must be 'x' or 'y'")

    if len(time_s) < n_pairs:
        raise ValueError("time_s must have length >= n_pairs")

    # Geometry: compute window centres using the same split_n_shift logic.
    # If ds_fac>1, match correlation's downsampled geometry and scale centres back.
    img = np.asarray(image)
    if img.ndim != 2:
        raise ValueError("image must be a 2D array")

    ds = int(ds_fac)
    if ds < 1:
        raise ValueError("ds_fac must be >= 1")

    if ds > 1:
        # `downsample` expects a stack; this matches `correlation.calc_corrs`.
        from tcm_piv.preprocessing import downsample

        img_ds = downsample(img[np.newaxis, ...], ds)[0]
    else:
        img_ds = img

    _, win_pos = split_n_shift(
        img_ds,
        (n_y, n_x),
        overlap=float(overlap),
        shift=(0, 0),
        shift_mode="before",
        plot=False,
    )

    # Use x-centres for the "distance from centre" axis.
    x_centres_px_ds = win_pos[..., 1]  # (n_y, n_x)
    x_centres_px = x_centres_px_ds * ds
    x0_px = img.shape[1] / 2.0
    dist_m = (x_centres_px - x0_px) * float(scale_m_per_px)

    # Collapse y dimension: profile vs x index.
    dist_profile = np.nanmean(dist_m, axis=0)  # (n_x,)
    order = np.argsort(dist_profile)
    dist_profile = dist_profile[order]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        for pair_i in range(n_pairs):
            stream = np.nanmean(vel_final[pair_i, :, :, stream_idx], axis=0)[order]
            cross = np.nanmean(vel_final[pair_i, :, :, cross_idx], axis=0)[order]

            fig, ax = plt.subplots(figsize=(6.5, 6.5))
            ax.plot(stream, dist_profile, "-o", label=stream_label, markersize=3)
            ax.plot(cross, dist_profile, "-o", label=cross_label, markersize=3)

            ax.set_xlabel("Velocity (m/s)")
            ax.set_ylabel("Distance from centre (m)")
            ax.set_title(f"Pair {pair_i} (t={float(time_s[pair_i]):.6g} s)")
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="best")

            pdf.savefig(fig)
            plt.close(fig)
