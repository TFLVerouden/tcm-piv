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
    mode: str | list[str] = "semicircle_rect",
    flow_direction: str = "x",
    output_path: Path | None = None,
) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.axhline(0.0, color="0.7", linewidth=1.0)
    ax.axvline(0.0, color="0.7", linewidth=1.0)

    if isinstance(mode, list):
        if len(mode) == 1 and ranges:
            mode_list = [str(mode[0]).strip().lower() for _ in ranges]
        elif len(mode) != len(ranges):
            raise ValueError(
                "mode must be a string or a list with length 1 or matching len(ranges)"
            )
        else:
            mode_list = [str(m).strip().lower() for m in mode]
    else:
        mode_list = [str(mode).strip().lower() for _ in ranges]

    mode_norm = mode_list[0] if mode_list else str(mode).strip().lower()
    flow_dir = str(flow_direction).strip().lower()
    if flow_dir not in {"x", "y"}:
        raise ValueError("flow_direction must be 'x' or 'y'")

    modes_unique = sorted(set(mode_list))
    title_mode = modes_unique[0] if len(modes_unique) == 1 else "mixed"

    for idx, (vx, vy) in enumerate(ranges, start=1):
        pass_mode = mode_list[idx - 1] if idx - \
            1 < len(mode_list) else mode_norm
        if pass_mode == "semicircle_rect":
            # Matches `filter_outliers('semicircle_rect', ...)`:
            # left half: circle of radius a (= vy) for vx < 0
            # right half: rectangle with vx in [0, b] (= vx) and vy in [-a, a]
            if flow_dir == "x":
                # streamwise = vx (x-axis), cross-stream = vy (y-axis)
                cross_limit = float(vy)
                stream_limit = float(vx)
                ax.add_patch(
                    Wedge(
                        (0.0, 0.0),
                        r=cross_limit,
                        theta1=90.0,
                        theta2=270.0,
                        fill=False,
                        linewidth=1.5,
                        label=f"Pass {idx}",
                    )
                )
                ax.add_patch(
                    Rectangle(
                        (0.0, -cross_limit),
                        stream_limit,
                        2 * cross_limit,
                        fill=False,
                        linewidth=1.5,
                    )
                )
            else:
                # streamwise = vy (y-axis), cross-stream = vx (x-axis)
                cross_limit = float(vx)
                stream_limit = float(vy)
                ax.add_patch(
                    Wedge(
                        (0.0, 0.0),
                        r=cross_limit,
                        theta1=180.0,
                        theta2=360.0,
                        fill=False,
                        linewidth=1.5,
                        label=f"Pass {idx}",
                    )
                )
                ax.add_patch(
                    Rectangle(
                        (-cross_limit, 0.0),
                        2 * cross_limit,
                        stream_limit,
                        fill=False,
                        linewidth=1.5,
                    )
                )
        elif pass_mode == "circle":
            radius = float(max(abs(vx), abs(vy)))
            ax.add_patch(
                Circle(
                    (0.0, 0.0),
                    radius=radius,
                    fill=False,
                    linewidth=1.5,
                    label=f"Pass {idx}",
                )
            )
        else:
            rect_patch = Rectangle(
                (-vx, -vy),
                2 * vx,
                2 * vy,
                fill=False,
                linewidth=1.5,
                label=f"Pass {idx}",
            )
            ax.add_patch(rect_patch)

    if ranges:
        vx_max = max(abs(vx) for vx, _ in ranges)
        vy_max = max(abs(vy) for _, vy in ranges)
        lim = max(vx_max, vy_max)
        pad = lim * 0.1 if lim else 1.0
        ax.set_xlim(-(lim + pad), lim + pad)
        ax.set_ylim(-(lim + pad), lim + pad)

    ax.set_xlabel("v_x limit (m/s)")
    ax.set_ylabel("v_y limit (m/s)")
    ax.set_title(f"Global filter ranges ({title_mode}, flow={flow_dir})")
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


def plot_correlation_map(
    corr: np.ndarray,
    *,
    center_yx: tuple[int, int] | np.ndarray,
    title: str | None = None,
    output_path: Path | None = None,
) -> tuple[Figure, Axes]:
    """Plot a single correlation map with a marker at the provided center.

    Args:
        corr: 2D correlation array.
        center_yx: (y, x) index of the zero-displacement reference point.
        title: Optional plot title.
        output_path: If provided, saves the figure to this path.
    """

    corr = np.asarray(corr)
    if corr.ndim != 2:
        raise ValueError("corr must be a 2D array")

    center_arr = np.asarray(center_yx).reshape(-1)
    if center_arr.size != 2:
        raise ValueError("center_yx must have 2 elements (y, x)")
    cy, cx = int(center_arr[0]), int(center_arr[1])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="viridis", origin="upper")
    ax.plot(cx, cy, marker="+", markersize=14,
            markeredgewidth=2.0, color="red", alpha=0.5)
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    if title:
        ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        plt.close(fig)

    return fig, ax


def export_velocity_profiles_pdf(
    *,
    vel_final: np.ndarray,
    interpolated_mask: np.ndarray | None = None,
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
    if interpolated_mask is not None:
        interpolated_mask = np.asarray(interpolated_mask)
        if interpolated_mask.shape != (n_pairs, n_y, n_x):
            raise ValueError(
                "interpolated_mask must have shape (n_pairs, n_y, n_x) matching vel_final"
            )

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
    image_arr = np.asarray(image)
    if image_arr.ndim != 2:
        raise ValueError("image must be a 2D array")

    downsample_factor = int(ds_fac)
    if downsample_factor < 1:
        raise ValueError("ds_fac must be >= 1")

    if downsample_factor > 1:
        # `downsample` expects a stack; this matches `correlation.calc_corrs`.
        from tcm_piv.preprocessing import downsample

        image_downsampled = downsample(image_arr[np.newaxis, ...], downsample_factor)[0]
    else:
        image_downsampled = image_arr

    _, win_pos = split_n_shift(
        image_downsampled,
        (n_y, n_x),
        overlap=float(overlap),
        shift=(0, 0),
        shift_mode="before",
        plot=False,
    )

    # Use x-centres for the "distance from centre" axis.
    x_centres_px_ds = win_pos[..., 1]  # (n_y, n_x)
    x_centres_px = x_centres_px_ds * downsample_factor
    x_center_px = image_arr.shape[1] / 2.0
    distances_m = (x_centres_px - x_center_px) * float(scale_m_per_px)

    # Collapse y dimension: profile vs x index.
    distance_profile_m = np.nanmean(distances_m, axis=0)  # (n_x,)
    sort_indices = np.argsort(distance_profile_m)
    distance_profile_m = distance_profile_m[sort_indices]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(output_path) as pdf:
        for pair_i in range(n_pairs):
            stream = np.nanmean(
                vel_final[pair_i, :, :, stream_idx], axis=0)[sort_indices]
            cross = np.nanmean(
                vel_final[pair_i, :, :, cross_idx], axis=0)[sort_indices]

            interpolated_columns = None
            if interpolated_mask is not None:
                # Mark a column if any y-position in that column was interpolated.
                interpolated_columns = np.any(
                    interpolated_mask[pair_i, :, :], axis=0)[sort_indices]

            fig, ax = plt.subplots(figsize=(6.5, 6.5))
            ax.plot(stream, distance_profile_m, "-o",
                    label=stream_label, markersize=3)
            ax.plot(cross, distance_profile_m, "-o",
                    label=cross_label, markersize=3)

            if interpolated_columns is not None and np.any(interpolated_columns):
                ax.plot(
                    stream[interpolated_columns],
                    distance_profile_m[interpolated_columns],
                    linestyle="none",
                    marker="x",
                    markersize=6,
                    color="crimson",
                    label="interpolated",
                )

            ax.set_xlabel("Velocity (m/s)")
            ax.set_ylabel("Distance from centre (m)")
            ax.set_title(f"Pair {pair_i} (t={float(time_s[pair_i]):.6g} s)")
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(loc="best")

            pdf.savefig(fig)
            plt.close(fig)
