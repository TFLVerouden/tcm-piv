"""Step-by-step PIV tuning tool for exactly one image pair.

Unlike `tcm_piv.run`, this script does not write any checkpoints and only
processes a single frame pair. It is meant for interactively tuning the
settings of one pass (crop/background, window layout, correlation,
peak-finding, global filter, neighbour filter) by inspecting a plot after
every step.

Usage:
    python scripts/tune_pass.py path/to/config.toml --pass 2 --pair-index 0

- `--pass` selects the 1-based pass index whose parameters to use (defaults
  to 1). Only that pass's settings are exercised; earlier passes are not
  run, so window shifts are always zero (no coarse-pass refinement).
- `--pair-index` selects which consecutive pair from the config's resolved
  `image_list` to load (0 -> first two images, 1 -> second and third, ...).
- Every step's plot is saved under `--output-dir` (default: a `tuning/`
  folder next to the config file). Pass `--show` to also open windows
  interactively.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

import tcm_piv as piv
from tcm_piv.init_config import Config, load_config
from tcm_piv.preprocessing import split_n_shift, denoise
from tcm_piv.run import _apply_crop_and_background, _neighbour_filter_strategy
from tcm_piv.visualisation import plot_correlation_map, plot_window_layout
from tcm_utils.io_utils import load_images

MAX_CORR_WINDOWS_TO_PLOT = 64
CORR_ZOOM_RADIUS_PX = 100

# Local/dev convenience: edit these and just press "Run" to skip the CLI.
# Any of these are overridden by passing the matching CLI argument instead.
DEFAULT_CONFIG_FILE = "/Volumes/Data/PIV/260820_piv/step_1-5bar/260831_152310_step_1-5bar_20-0mA/P-001/piv/config_piv_diagnostic.toml"
DEFAULT_PASS_1B = 1
DEFAULT_PAIR_INDEX = 0


def auto_brightness(image: np.ndarray, low_percentile: float = 0.5, high_percentile: float = 99.95) -> np.ndarray:
    """Contrast-stretch an image to uint8 using percentile clipping."""
    arr = np.asarray(image, dtype=np.float64)
    lo, hi = np.percentile(arr, [low_percentile, high_percentile])
    if hi <= lo:
        hi = lo + 1.0
    stretched = np.clip((arr - lo) / (hi - lo), 0.0, 1.0) * 255.0
    return stretched.astype(np.uint8)


def _pass_params(config: Config, pass_idx0: int) -> dict:
    """Pull out the settings for a single pass from a (possibly multi-pass) config."""
    return dict(
        ds_factor=int(config.ds_factor[pass_idx0]),
        n_windows=config.n_windows[pass_idx0],
        overlap=float(config.window_overlap[pass_idx0]),
        n_peaks=int(config.n_peaks[pass_idx0]),
        min_peak_dist_px=int(config.min_peak_dist_px[pass_idx0]),
        outlier_mode=str(
            config.outlier_filter_mode[pass_idx0]).strip().lower(),
        vy_max_m_s=float(config.max_velocity_vy_vx_m_s[pass_idx0][0]),
        vx_max_m_s=float(config.max_velocity_vy_vx_m_s[pass_idx0][1]),
        nb_size_tyx=config.nb_size_tyx[pass_idx0],
        nb_threshold=config.nb_threshold[pass_idx0],
    )


def _global_filter(
    disp_peaks: np.ndarray,
    *,
    outlier_mode: str,
    flow_direction: str,
    vy_max_m_s: float,
    vx_max_m_s: float,
    timestep_s: float,
    scale_m_per_px: float,
) -> np.ndarray:
    """Mirror `tcm_piv.run`'s global-filter axis handling for one pass."""

    a_y = vy_max_m_s * timestep_s / scale_m_per_px
    b_x = vx_max_m_s * timestep_s / scale_m_per_px
    a_x = vx_max_m_s * timestep_s / scale_m_per_px
    b_y = vy_max_m_s * timestep_s / scale_m_per_px

    if outlier_mode == "semicircle_rect":
        if flow_direction == "x":
            a_px, b_px, disp_for_filter, unswap = a_y, b_x, disp_peaks, False
        else:
            a_px, b_px, disp_for_filter, unswap = a_x, b_y, disp_peaks[..., [
                1, 0]], True
    elif outlier_mode == "circle":
        a_px = max(abs(vx_max_m_s), abs(vy_max_m_s)) * \
            timestep_s / scale_m_per_px
        b_px, disp_for_filter, unswap = None, disp_peaks, False
    else:
        raise ValueError(
            "outlier_filter_mode must be 'semicircle_rect' or 'circle'")

    disp_filtered = np.asarray(
        piv.filter_outliers(outlier_mode, disp_for_filter,
                            a=a_px, b=b_px, verbose=True)
    )
    return disp_filtered[..., [1, 0]] if unswap else disp_filtered


def _plot_frame_pair(imgs: np.ndarray, title: str, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), constrained_layout=True)
    for ax, img, label in zip(axes, imgs, ("Frame 0", "Frame 1")):
        ax.imshow(auto_brightness(img), cmap="gray", vmin=0, vmax=255)
        ax.set_title(label)
        ax.set_axis_off()
    fig.suptitle(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)


def _peak_yx_in_corr(map_center: np.ndarray, disp_yx: np.ndarray, ds_factor: int) -> np.ndarray:
    """Invert `find_disp`'s displacement formula to get a peak's (y, x) in corr-map coords."""
    return np.asarray(map_center, dtype=np.float64) + np.asarray(disp_yx, dtype=np.float64) / ds_factor


def _plot_correlation_grid(
    corrs: dict,
    disp_peaks: np.ndarray,
    n_windows: tuple[int, int],
    ds_factor: int,
    zoom_radius_px: int,
    output_path: Path,
) -> None:
    n_wy, n_wx = n_windows
    if n_wy * n_wx > MAX_CORR_WINDOWS_TO_PLOT:
        print(
            f"  Skipping correlation grid: {n_wy * n_wx} windows exceeds "
            f"{MAX_CORR_WINDOWS_TO_PLOT} (use plot_correlation_map manually for a single window)."
        )
        return
    n_peaks = disp_peaks.shape[3]
    fig, axes = plt.subplots(
        n_wy, n_wx, figsize=(2.2 * n_wx, 2.2 * n_wy), squeeze=False)
    for j in range(n_wy):
        for k in range(n_wx):
            corr_map, center = corrs[(0, j, k)]
            corr_map = np.asarray(corr_map)
            cy, cx = int(center[0]), int(center[1])

            y0 = max(0, cy - zoom_radius_px)
            y1 = min(corr_map.shape[0], cy + zoom_radius_px)
            x0 = max(0, cx - zoom_radius_px)
            x1 = min(corr_map.shape[1], cx + zoom_radius_px)

            ax = axes[j][k]
            ax.imshow(corr_map[y0:y1, x0:x1], cmap="viridis",
                      extent=(x0, x1, y1, y0))
            ax.plot(cx, cy, "+", color="red", markersize=8,
                    label="Zero displacement")

            for p in range(n_peaks):
                peak_yx = _peak_yx_in_corr(
                    center, disp_peaks[0, j, k, p, :], ds_factor)
                if np.any(np.isnan(peak_yx)):
                    continue
                is_best = p == 0
                ax.plot(
                    peak_yx[1], peak_yx[0],
                    marker="o",
                    markersize=6 if is_best else 3,
                    markerfacecolor="none" if is_best else "tab:orange",
                    markeredgecolor="tab:orange" if is_best else "none",
                    markeredgewidth=1.5,
                )

            ax.set_xlim(x0, x1)
            ax.set_ylim(y1, y0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"({j},{k})", fontsize=7)
    fig.suptitle(
        f"Correlation maps per window (zoom \u00b1{zoom_radius_px}px)\n"
        "best peak = open circle, others = filled",
        fontsize=8,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)


def _plot_quiver(
    image: np.ndarray,
    win_pos: np.ndarray,
    disp_yx: np.ndarray,
    title: str,
    output_path: Path,
    *,
    candidates_yx: np.ndarray | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(auto_brightness(image), cmap="gray", vmin=0, vmax=255)

    y = win_pos[..., 0].ravel()
    x = win_pos[..., 1].ravel()
    dy = disp_yx[..., 0].ravel()
    dx = disp_yx[..., 1].ravel()
    valid = ~np.isnan(dy) & ~np.isnan(dx)

    if candidates_yx is not None:
        for p in range(candidates_yx.shape[-2]):
            cdy = candidates_yx[..., p, 0].ravel()
            cdx = candidates_yx[..., p, 1].ravel()
            cvalid = ~np.isnan(cdy) & ~np.isnan(cdx)
            ax.quiver(
                x[cvalid], y[cvalid], cdx[cvalid], cdy[cvalid],
                angles="xy", scale_units="xy", scale=0.2,
                color="tab:orange", alpha=0.4, width=0.0025, zorder=2,
                label="Other candidate peaks" if p == 0 else None,
            )

    ax.quiver(x[valid], y[valid], dx[valid], dy[valid],
              angles="xy", scale_units="xy", scale=0.2, color="red",
              zorder=3, label="Best peak" if candidates_yx is not None else None)
    ax.set_title(f"{title} ({int(valid.sum())}/{valid.size} valid)")
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    if candidates_yx is not None:
        ax.legend(loc="upper right")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step-by-step PIV tuning on a single image pair.")
    parser.add_argument("config_file", nargs="?", default=DEFAULT_CONFIG_FILE,
                        help="Path to a TOML config file.")
    parser.add_argument("--pass", dest="pass_1b", type=int, default=DEFAULT_PASS_1B,
                        help="1-based pass index whose settings to use.")
    parser.add_argument("--pair-index", type=int, default=DEFAULT_PAIR_INDEX,
                        help="0-based index into the resolved image_list to pick a pair.")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save step plots (default: tuning/ next to the config).")
    parser.add_argument("--corr-zoom-px", type=int, default=CORR_ZOOM_RADIUS_PX,
                        help="Half-width (in px) of the zoomed region shown around each correlation map's centre.")
    parser.add_argument("--show", action="store_true",
                        help="Also display the plots interactively.")
    args = parser.parse_args()

    config, config_path = load_config(args.config_file)
    pass_idx0 = args.pass_1b - 1
    if not (0 <= pass_idx0 < config.nr_passes):
        raise ValueError(
            f"--pass {args.pass_1b} is out of range for nr_passes={config.nr_passes}")

    if args.pair_index + 1 >= config.nr_images:
        raise ValueError(
            f"--pair-index {args.pair_index} needs images "
            f"[{args.pair_index}, {args.pair_index + 1}] but only {config.nr_images} are resolved."
        )

    output_dir = Path(args.output_dir) if args.output_dir else (
        config.config_path.parent / "tuning" /
        f"pass{args.pass_1b:02d}_pair{args.pair_index:04d}"
    )
    print(f"Output directory: {output_dir}")

    params = _pass_params(config, pass_idx0)
    print(f"Pass {args.pass_1b} settings: {params}")

    # Step 1: load the two raw frames.
    paths = [config.image_list[args.pair_index],
             config.image_list[args.pair_index + 1]]
    print(f"Step 1: loading frames: {[Path(p).name for p in paths]}")
    imgs_raw = np.asarray(load_images(paths, show_progress=False))
    _plot_frame_pair(imgs_raw, "Step 1: raw frames",
                     output_dir / "01_raw_frames.png")

    # Step 2: crop + background subtraction (as configured).
    print("Step 2: applying crop + background subtraction")
    imgs = _apply_crop_and_background(imgs_raw, config)
    # imgs = denoise(imgs, lower_intensity=2, upper_intensity=4096,
    #                shift_to_zero=True, reduce_dtype=True)
    _plot_frame_pair(imgs, "Step 2: after crop + background",
                     output_dir / "02_preprocessed_frames.png")

    # Step 3: window layout.
    print(f"Step 3: window layout n_windows={params['n_windows']}")
    plot_window_layout(
        auto_brightness(imgs[0]),
        params["n_windows"],
        overlap=params["overlap"],
        title=f"Pass {args.pass_1b} windows",
        output_path=output_dir / "03_window_layout.png",
    )
    plt.close("all")

    # Step 4: correlation maps (no shifts: this pass is tested in isolation).
    print(
        f"Step 4: calculating correlation maps (ds_factor={params['ds_factor']})")
    corrs = piv.calc_corrs(
        imgs,
        n_windows=params["n_windows"],
        shifts=None,
        overlap=params["overlap"],
        ds_factor=params["ds_factor"],
    )

    # Step 5: peak-finding (multi-peak candidates).
    print(f"Step 5: finding peaks (n_peaks={params['n_peaks']})")
    disp_peaks, peak_int = piv.find_disps(
        corrs,
        n_windows=params["n_windows"],
        shifts=None,
        n_peaks=params["n_peaks"],
        ds_factor=params["ds_factor"],
        min_dist=params["min_peak_dist_px"],
        do_subpixel=True,
    )
    _plot_correlation_grid(
        corrs, disp_peaks, params["n_windows"], params["ds_factor"],
        args.corr_zoom_px, output_dir / "04_correlation_maps.png",
    )
    plt.close("all")

    _, win_pos = split_n_shift(
        imgs[0], params["n_windows"], overlap=params["overlap"],
        shift=(0, 0), plot=False,
    )
    # (1, n_wy, n_wx, 2), best peak only
    disp_best_raw = disp_peaks[:, :, :, 0, :]
    candidates_raw = disp_peaks[0, :, :, 1:,
                                :] if params["n_peaks"] > 1 else None
    _plot_quiver(imgs[0], win_pos, disp_best_raw[0], "Step 5: best-peak displacement (unfiltered)",
                 output_dir / "05_raw_peaks_quiver.png", candidates_yx=candidates_raw)
    plt.close("all")

    # Step 6: global (velocity-bound) outlier filter.
    print(
        f"Step 6: global filter mode={params['outlier_mode']} "
        f"vy_max={params['vy_max_m_s']} vx_max={params['vx_max_m_s']} m/s"
    )
    disp_global_5d = _global_filter(
        disp_peaks,
        outlier_mode=params["outlier_mode"],
        flow_direction=config.flow_direction,
        vy_max_m_s=params["vy_max_m_s"],
        vx_max_m_s=params["vx_max_m_s"],
        timestep_s=config.timestep_s,
        scale_m_per_px=config.scale_m_per_px,
    )
    disp_global = piv.strip_peaks(
        disp_global_5d, axis=-2, mode="reduce", verbose=True)
    _plot_quiver(imgs[0], win_pos, disp_global[0], "Step 6: after global filter",
                 output_dir / "06_global_filter_quiver.png")
    plt.close("all")

    # Step 7: neighbour filter (spatial only; there is just one time step).
    nb_size_tyx = params["nb_size_tyx"]
    if nb_size_tyx[0] > 1:
        print(
            f"  Note: nb_size_tyx time extent ({nb_size_tyx[0]}) is ignored "
            "with only one frame pair; using 1 instead."
        )
    nb_size_effective = (1, nb_size_tyx[1], nb_size_tyx[2])
    nb_mode, nb_replace, nb_thr_unit = _neighbour_filter_strategy(
        pass_idx0, params["nb_threshold"])
    print(
        f"Step 7: neighbour filter mode={nb_mode} threshold_unit={nb_thr_unit} "
        f"replace={nb_replace} neighbourhood={nb_size_effective}"
    )
    disp_nb = piv.filter_neighbours(
        disp_global,
        threshold=params["nb_threshold"],
        threshold_unit=nb_thr_unit,
        neighbourhood_size=nb_size_effective,
        mode=nb_mode,
        replace=nb_replace,
        verbose=True,
    )
    _plot_quiver(imgs[0], win_pos, disp_nb[0], "Step 7: after neighbour filter",
                 output_dir / "07_neighbour_filter_quiver.png")
    plt.close("all")

    # Step 8: physical velocity summary.
    vel_final = disp_nb[0] * config.scale_m_per_px / \
        config.timestep_s  # (n_wy, n_wx, 2)
    vy, vx = vel_final[..., 0], vel_final[..., 1]
    print("Step 8: final velocity summary (m/s)")
    print(
        f"  vy: median={np.nanmedian(vy):.4g}, min={np.nanmin(vy):.4g}, max={np.nanmax(vy):.4g}")
    print(
        f"  vx: median={np.nanmedian(vx):.4g}, min={np.nanmin(vx):.4g}, max={np.nanmax(vx):.4g}")
    print(f"  valid windows: {int(np.count_nonzero(~np.isnan(vy)))}/{vy.size}")

    if args.show:
        plt.show()

    print(f"\nDone. Plots saved under: {output_dir}")


if __name__ == "__main__":
    main()
