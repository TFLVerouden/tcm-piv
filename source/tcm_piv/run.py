"""CLI entrypoint for tcm-piv.

This file orchestrates the end-to-end PIV pipeline:
- Read/normalize a TOML config via :func:`tcm_piv.init_config.load_config`.
- Create a run directory under `<output_dir>/runs/<run_id>/` unless the
    config sets ``overwrite_runs = true``, in which case results are written
    directly into ``output_dir``.
- For each pass, run correlation + peak finding + postprocessing and write:

    - `pass_XX_post.csv`: the final single-peak displacement field plus some
        intermediate columns.
    - `pass_XX_meta.json`: small metadata describing shapes/config for the pass.

The mapping from `pair_index -> image0,image1` is written once in `pairs.csv`
in the run root.

---

How to run
----------

This module is designed to be run as a module:

`python -m tcm_piv.run path/to/config.toml`

The config controls both the *input* (image directory + camera/calibration
metadata) and the *output directory* where runs are created.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import cv2 as cv
import numpy as np

import tcm_piv as piv
import tcm_piv.visualisation as viz
from tcm_piv.preprocessing import crop, denoise_images
from tcm_utils.cough_model import CoughModel
from tcm_piv.outputs import (
    init_run_dir,
    pass_paths,
    write_meta_json,
    write_pairs_csv,
    write_postprocessed_csv,
)
from tcm_utils.io_utils import load_images
from tcm_utils.time_utils import timestamp_str
from tcm_utils.cvd_check import set_cvd_friendly_colors
from tcm_piv.init_config import Config, load_config, archive_config


def run(
    *,
    config_file: str | Path | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> Path:
    """Run the PIV pipeline.

    Args:
        config_file: Path to a TOML config file. If None, a file picker prompt is used.

    Returns:
        run_dir (Path): Path to the run directory where results are stored.
    """

    print("\n\nStarting PIV analysis...")
    config_path = Path(config_file) if config_file else None

    if config_path is not None:
        print(f"Config file: {config_path}")

    # Step 0: Load + normalize configuration (TOML + defaults + runtime resolution).
    print("\nReading config...")
    config, config_path = load_config(config_path, overrides=config_overrides)

    print("Config summary:")
    print(f"  image_dir: {config.image_dir}")
    print(f"  nr_images: {config.nr_images}")
    print(f"  nr_passes: {config.nr_passes}")
    print(f"  timestep_s: {config.timestep_s}")
    print(f"  scale_m_per_px: {config.scale_m_per_px}")
    print(f"  output_dir: {config.output_dir}")
    print(
        f"  overwrite_runs: {bool(getattr(config, 'overwrite_runs', False))}")

    # Each run gets its own directory so results are reproducible.
    # Step 1: Create a run directory.
    run_id = timestamp_str()
    run_dir = init_run_dir(
        config.output_dir,
        run_id,
        overwrite_runs=bool(getattr(config, "overwrite_runs", False)),
    )
    print(f"\nRun directory: {run_dir}")

    # Archive the configuration file.
    archive_config(config_path, run_dir)

    # Pair mapping for CSV row -> image filenames.
    #
    # All result CSVs refer to images by *pair_index* (0-based), not by
    # embedding filenames in every row. `pairs.csv` is the lookup table.
    # This keeps large CSV outputs compact.

    # Step 2: Write pairs.csv (pair_index -> image filenames).
    pairs = [
        (
            i,
            _relpath_or_name(
                config.image_list[i], base_dir=str(config.image_dir)),
            _relpath_or_name(
                config.image_list[i + 1], base_dir=str(config.image_dir)),
        )
        for i in range(config.nr_images - 1)
    ]
    print(f"Writing pairs.csv ({len(pairs)} pairs)...")
    write_pairs_csv(run_dir / "pairs.csv", pairs)

    # Timebase used for plotting / any temporal processing.
    # `time_s` has length (NR_IMAGES - 1), matching the number of image pairs.
    # Step 3: Build the timebase (one timestamp per image pair).
    if np.any(config.frames_to_use == 0):
        raise ValueError(
            "frames_to_use must be 1-based (0 is not a valid frame number).")
    frames = config.frames_to_use if isinstance(config.frames_to_use, list) else list(
        range(1, config.nr_images + 1)
    )
    print(f"Timebase: {len(frames)} frames -> {len(frames) - 1} pairs")
    time_s = piv.get_time(frames, float(config.timestep_s))

    # For plotting filter ranges we keep (vx, vy) ordering.
    filter_ranges_vx_vy: list[tuple[float, float]] = []
    window_layouts: list[dict[str, Any]] = []
    # (vector overlay plotting removed for now)
    disp_final_lastpass: np.ndarray | None = None
    interp_filled_mask: np.ndarray | None = None

    # Lazily loaded image stack. We only load if we need correlation.
    imgs: np.ndarray | None = None

    # Previous pass output (postprocessed) is used to compute shifts for the
    # next pass (multi-pass PIV refinement).
    prev_disp_final: np.ndarray | None = None

    # Step 4: Multi-pass refinement loop.
    for pass_idx0 in range(config.nr_passes):
        pass_idx1 = pass_idx0 + 1
        paths = pass_paths(run_dir, pass_idx1)

        plot_windows = bool(config.plot_window_layout[pass_idx0])
        # vector overlays removed for now

        n_win_y, n_win_x = config.n_windows[pass_idx0]
        n_img_pairs = len(time_s)

        print(
            f"\nPASS {pass_idx1:02d}/{int(config.nr_passes):02d}: n_windows=({n_win_y},{n_win_x})"
        )

        # Config ordering is always [vy, vx].
        vy_max_m_s, vx_max_m_s = config.max_velocity_vy_vx_m_s[pass_idx0]
        filter_ranges_vx_vy.append((float(vx_max_m_s), float(vy_max_m_s)))

        nb_thr = config.nb_threshold[pass_idx0]
        nb_size_tyx = config.nb_size_tyx[pass_idx0]
        interp_nb_size_tyx = config.interp_nb_size_tyx[pass_idx0]
        time_smooth_lam = float(config.time_smooth_lam[pass_idx0])

        ds_factor = int(config.ds_factor[pass_idx0])
        n_corrs_to_sum = int(config.n_corrs_to_sum[pass_idx0])
        n_peaks = int(config.n_peaks[pass_idx0])
        min_peak_dist_px = int(config.min_peak_dist_px[pass_idx0])
        overlap = float(config.window_overlap[pass_idx0])

        print("Pass parameters:")
        print(f"  downsample_factor: {ds_factor}")
        print(f"  corrs_to_sum: {n_corrs_to_sum}")
        print(f"  nr_peaks: {n_peaks}")
        print(f"  min_peak_distance_px: {min_peak_dist_px}")
        print(f"  window_overlap: {overlap}")
        print(f"  max_velocity (vy,vx) [m/s]: ({vy_max_m_s},{vx_max_m_s})")

        # Metadata is written once the pass completes.
        # It documents shapes, key config knobs, and the filenames that belong
        # to this pass.
        meta: dict[str, Any] = {
            "pass": pass_idx1,
            "n_pairs": n_img_pairs,
            "n_windows": [n_win_y, n_win_x],
            "downsample_factor": ds_factor,
            "corrs_to_sum": n_corrs_to_sum,
            "nr_peaks": n_peaks,
            "min_peak_distance": min_peak_dist_px,
            "window_overlap": overlap,
            "max_velocity_vy_vx_m_s": [float(vy_max_m_s), float(vx_max_m_s)],
            "max_velocity_vx_vy_m_s": [float(vx_max_m_s), float(vy_max_m_s)],
            "outlier_filter_mode": str(config.outlier_filter_mode[pass_idx0]).strip().lower(),
            "neighbourhood_size": list(nb_size_tyx),
            "neighbourhood_threshold": nb_thr,
            "time_smoothing_lambda": time_smooth_lam,
            "files": {
                "pairs": "../pairs.csv",
                "post": paths.post_csv.name,
            },
        }

        # Step 4b: Load images once (only if we need to compute correlations).
        if imgs is None:

            print("Loading images...")
            imgs = np.asarray(load_images(
                config.image_list, show_progress=True))

            # Step 4c: Pre-processing
            imgs = _apply_crop_and_background(imgs, config)
            imgs = denoise_images(imgs, background_blur_sigma=config.denoise_blur_sigma,
                                  lower_intensity=config.denoise_lower_int,
                                  upper_intensity=config.denoise_upper_int,
                                  shift_to_zero=True, reduce_dtype=True)

        # Later passes refine the search region by shifting windows based
        # on the previous pass result.
        if pass_idx0 == 0:
            shifts = None
        else:
            if prev_disp_final is None:
                raise RuntimeError(
                    f"Pass {pass_idx1} needs previous displacement to compute shifts")
            print("Computing window shifts from previous pass...")
            shifts = piv.disp2shift(
                n_windows=(n_win_y, n_win_x),
                displacements=prev_disp_final,
            )

            # Replace NaNs (from invalid vectors in prev pass) with 0.0 shift
            shifts = np.nan_to_num(shifts, nan=0.0)

        # 1) Calculate correlations per pair/window.
        print("Step 1: calculating correlation maps...")
        corrs = piv.calc_corrs(
            imgs,
            n_windows=(n_win_y, n_win_x),
            shifts=shifts,
            overlap=overlap,
            ds_factor=ds_factor,
        )

        # 2) Optionally sum correlations over a time window.
        #    This is a denoising/smoothing step in correlation space.
        print(
            f"Step 2: summing correlation maps (corrs_to_sum={n_corrs_to_sum})...")
        corrs_sum = piv.sum_corrs(
            corrs,
            n_corrs_to_sum,
            n_windows=(n_win_y, n_win_x),
            shifts=shifts,
        )

        if bool(config.plot_correlations):
            plots_dir = run_dir / "plots"
            corr_dir = plots_dir / "correlations" / f"pass_{pass_idx1:02d}"
            j_mid = int(n_win_y) // 2
            k_mid = int(n_win_x) // 2
            for pair_i in range(n_img_pairs):
                corr_map, corr_center = corrs_sum[(pair_i, j_mid, k_mid)]
                title = f"Pass {pass_idx1} pair {pair_i} win ({j_mid},{k_mid})"
                viz.plot_correlation_map(
                    np.asarray(corr_map),
                    center_yx=(int(corr_center[0]), int(corr_center[1])),
                    title=title,
                    output_path=corr_dir / f"pair_{pair_i:04d}.png",
                )

        # 3) Find displacement peaks in the correlation planes.
        print(f"Step 3: finding peaks (nr_peaks={n_peaks})...")
        disp_peaks_unf, peak_int_unf = piv.find_disps(
            corrs_sum,
            n_windows=(n_win_y, n_win_x),
            shifts=shifts,
            n_peaks=n_peaks,
            ds_factor=ds_factor,
            min_dist=min_peak_dist_px,
            do_subpixel=pass_idx0 == config.nr_passes - 1,
        )

        # Postprocess the multi-peak results into a usable single-peak
        # displacement field, applying outlier and neighbour filtering.
        print("Postprocessing: outlier + neighbour filtering")
        a_y = float(vy_max_m_s) * float(config.timestep_s) / \
            float(config.scale_m_per_px)
        b_x = float(vx_max_m_s) * float(config.timestep_s) / \
            float(config.scale_m_per_px)
        a_x = float(vx_max_m_s) * float(config.timestep_s) / \
            float(config.scale_m_per_px)
        b_y = float(vy_max_m_s) * float(config.timestep_s) / \
            float(config.scale_m_per_px)

        nb_mode, nb_replace, nb_thr_unit = _neighbour_filter_strategy(
            pass_idx0, nb_thr)

        outlier_mode = str(
            config.outlier_filter_mode[pass_idx0]).strip().lower()
        flow_dir = str(config.flow_direction).strip().lower()

        print(
            f"  outlier_filter_mode: {outlier_mode} (flow_direction={flow_dir})")

        if outlier_mode == "semicircle_rect":
            # `filter_outliers('semicircle_rect')` is asymmetric in the second
            # coordinate ("x"). Our displacement coords are (dy, dx).
            #
            # If flow is along x: keep as-is (streamwise=dx, cross-stream=dy).
            # If flow is along y: swap axes (streamwise=dy, cross-stream=dx).
            if flow_dir == "x":
                a_px: float | np.ndarray = a_y
                b_px: float | None = b_x
                disp_for_filter = disp_peaks_unf
                unswap: bool = False
            else:  # flow_dir == "y"
                a_px = a_x
                b_px = b_y
                disp_for_filter = disp_peaks_unf[..., [1, 0]]
                unswap = True
        elif outlier_mode == "circle":
            # Use the larger of the two maxima as the circle radius.
            vmax = float(max(abs(vx_max_m_s), abs(vy_max_m_s)))
            a_px = vmax * float(config.timestep_s) / \
                float(config.scale_m_per_px)
            b_px = None
            disp_for_filter = disp_peaks_unf
            unswap = False
        else:
            raise ValueError(
                "outlier_filter_mode must be 'semicircle_rect' or 'circle'")

        disp_global_5d = np.asarray(
            piv.filter_outliers(
                outlier_mode,
                disp_for_filter,
                a=a_px,
                b=b_px,
                verbose=True,
            )
        )
        if unswap:
            disp_global_5d = disp_global_5d[..., [1, 0]]

        if len(nb_size_tyx) != 3:
            raise ValueError(
                f"nb_size_tyx must have 3 elements, got {nb_size_tyx}")
        if len(interp_nb_size_tyx) != 3:
            raise ValueError(
                f"interp_nb_size_tyx must have 3 elements, got {interp_nb_size_tyx}")

        if nb_replace == "closest":
            print(
                f"  neighbour_filter: mode={nb_mode} threshold_unit={nb_thr_unit} replace={nb_replace}"
            )
            disp_nb_5d = piv.filter_neighbours(
                disp_global_5d,
                threshold=nb_thr,  # type: ignore[arg-type]
                threshold_unit=nb_thr_unit,
                neighbourhood_size=nb_size_tyx,
                mode=nb_mode,
                replace=nb_replace,
                verbose=True,
                timing=True,
            )

            disp_global = piv.strip_peaks(
                disp_global_5d, axis=-2, mode="reduce", verbose=False
            )
            disp_nb = piv.strip_peaks(
                disp_nb_5d, axis=-2, mode="reduce", verbose=True
            )
            disp_final = disp_nb
        else:
            print(
                f"  neighbour_filter: mode={nb_mode} threshold_unit={nb_thr_unit} replace={nb_replace}"
            )
            disp_global = piv.strip_peaks(
                disp_global_5d, axis=-2, mode="reduce", verbose=True
            )
            disp_nb = piv.filter_neighbours(
                disp_global,
                threshold=nb_thr,  # type: ignore[arg-type]
                threshold_unit=nb_thr_unit,
                neighbourhood_size=nb_size_tyx,
                mode=nb_mode,
                replace=nb_replace,
                verbose=True,
                timing=True,
            )
            disp_final = disp_nb

        disp_final_presmooth: np.ndarray | None = None
        if time_smooth_lam and time_smooth_lam > 0:
            if disp_final.shape[0] >= 3 and disp_final.shape[1:3] == (1, 1):
                print(
                    f"  temporal_smoothing: enabled (lambda={time_smooth_lam})"
                )
                disp_final_presmooth = disp_final.copy()
                disp_final = piv.smooth(
                    time_s,
                    disp_final,
                    smoothing_lambda=time_smooth_lam,
                    dtype=int,
                )

        # Single-window passes: always plot the (unaveraged) displacement
        # time series, since there is only one value per pair and no
        # meaningful median/quantile to take across windows.
        if disp_final.shape[1:3] == (1, 1):
            plots_dir = run_dir / "plots"
            if disp_final_presmooth is not None:
                raw_for_plot = disp_final_presmooth
                raw_label, final_label = "Raw", "Smoothed"
                title = f"Pass {pass_idx1} displacements: raw vs smoothed (lambda={time_smooth_lam})"
            else:
                raw_for_plot = disp_global
                raw_label, final_label = "Before neighbour filter", "Final"
                title = f"Pass {pass_idx1} displacements"
            viz.plot_temporal_displacements(
                time_s,
                raw_for_plot,
                disp_final,
                title=title,
                raw_label=raw_label,
                final_label=final_label,
                output_path=plots_dir
                / "displacements"
                / f"pass_{pass_idx1:02d}_temporal_displacements.png",
            )

        # For multi-window passes, summarize displacements over windows per timestamp.
        elif disp_final.ndim == 4 and disp_final.shape[-1] == 2:
            plots_dir = run_dir / "plots"
            viz.plot_temporal_displacement_quantiles(
                time_s,
                disp_final,
                title=f"Pass {pass_idx1} displacements over windows (Q1/median/Q3)",
                output_path=plots_dir
                / "displacements"
                / f"pass_{pass_idx1:02d}_window_quantiles.png",
            )

        # Final pass: patch remaining NaN holes by interpolation
        if (
            pass_idx0 == config.nr_passes - 1
            and any(v > 1 for v in interp_nb_size_tyx)
            and np.isnan(disp_final).any()
        ):
            print(
                f"  interpolation: patching NaN holes (neighbourhood_size={interp_nb_size_tyx})"
            )
            disp_pre_interp = disp_final
            nan_mask = np.any(np.isnan(disp_pre_interp), axis=-1)

            disp_patched = piv.filter_neighbours(
                disp_pre_interp,
                threshold=None,
                neighbourhood_size=interp_nb_size_tyx,
                mode="xy",
                replace="interp",
                verbose=True,
                timing=True,
            )
            filled_mask = nan_mask & (~np.any(np.isnan(disp_patched), axis=-1))
            interp_filled_mask = filled_mask
            disp_final = disp_patched

        if plot_windows:
            window_layouts.append({
                "pass": pass_idx1,
                "n_windows": (n_win_y, n_win_x),
                "overlap": overlap,
                "shifts": shifts,
            })

        if pass_idx0 == config.nr_passes - 1:
            disp_final_lastpass = disp_final

        print(f"Writing pass results: {paths.post_csv.name}")
        write_postprocessed_csv(
            paths.post_csv,
            time_s=time_s,
            disp_glo=disp_global,
            disp_nbs=disp_nb,
            disp_final=disp_final,
        )
        prev_disp_final = disp_final

        # Remove some stuff from memory
        del corrs, corrs_sum, disp_peaks_unf, peak_int_unf, disp_global_5d, disp_global, disp_nb, disp_final, shifts
        gc.collect()

    if disp_final_lastpass is not None:
        print("\nFinal exports: velocity + flow rate")
        vel_final = disp_final_lastpass * \
            float(config.scale_m_per_px) / float(config.timestep_s)
        flow_m3s = piv.vel2flow(
            vel_final,
            float(config.extra_vel_dim_m),
            float(config.image_width_m),
            float(config.image_height_m),
            flow_direction=str(config.flow_direction),
        )
        flow_ls = flow_m3s * 1000.0

        n_pairs, n_wy, n_wx, _ = vel_final.shape
        pair_index = np.repeat(np.arange(n_pairs, dtype=np.int64), n_wy * n_wx)
        win_y = np.tile(
            np.repeat(np.arange(n_wy, dtype=np.int64), n_wx), n_pairs)
        win_x = np.tile(np.arange(n_wx, dtype=np.int64), n_pairs * n_wy)
        time_rep = np.repeat(time_s[:n_pairs], n_wy * n_wx)

        vel_csv = run_dir / "velocity_final.csv"
        print(f"Writing: {vel_csv.name}")
        np.savetxt(
            vel_csv,
            np.column_stack([
                pair_index,
                win_y,
                win_x,
                time_rep,
                vel_final[..., 0].reshape(-1),
                vel_final[..., 1].reshape(-1),
            ]),
            delimiter=",",
            header="pair_index,win_y,win_x,time_s,vy_m_s,vx_m_s",
            comments="",
        )

        if interp_filled_mask is not None:
            interp_csv = run_dir / "interpolated_mask.csv"
            print(f"Writing: {interp_csv.name}")
            interp_flat = interp_filled_mask.reshape(-1).astype(np.int64)
            np.savetxt(
                interp_csv,
                np.column_stack([
                    pair_index,
                    win_y,
                    win_x,
                    interp_flat,
                ]),
                delimiter=",",
                header="pair_index,win_y,win_x,interpolated",
                comments="",
            )

        flow_csv = run_dir / "flow_rate.csv"
        print(f"Writing: {flow_csv.name}")
        np.savetxt(
            flow_csv,
            np.column_stack(
                [np.arange(len(flow_ls)), time_s[: len(flow_ls)],
                 flow_m3s, flow_ls]
            ),
            delimiter=",",
            header="pair_index,time_s,flow_rate_m3_s,flow_rate_L_s",
            comments="",
        )

        plots_dir = run_dir / "plots"
        if config.plot_global_filters and filter_ranges_vx_vy:
            print("Plotting: filter_ranges.png")
            viz.plot_filter_ranges(
                filter_ranges_vx_vy,
                mode=config.outlier_filter_mode,
                flow_direction=str(config.flow_direction),
                output_path=plots_dir / "filter_ranges.png",
            )

        image_for_plots: np.ndarray | None = None
        if window_layouts or config.export_velocity_profiles_pdf:
            if imgs is not None:
                image_for_plots = np.asarray(imgs[0])
            else:
                base = load_images([config.image_list[0]], show_progress=False)
                image_for_plots = np.asarray(base[0])
                image_for_plots = _apply_crop_and_background(
                    image_for_plots, config)

        if window_layouts and image_for_plots is not None:
            print("Plotting: window layouts")
            for layout in window_layouts:
                viz.plot_window_layout(
                    image_for_plots,
                    layout["n_windows"],
                    overlap=float(layout.get("overlap", 0.0)),
                    shifts=layout.get("shifts"),
                    shift_mode="before",
                    title=f"Pass {layout['pass']:02d} windows",
                    output_path=plots_dir /
                    f"pass_{layout['pass']:02d}_windows.png",
                )

        if config.export_velocity_profiles_pdf and image_for_plots is not None:
            print("Exporting: velocity_profiles.pdf")
            final_pass_i = int(config.nr_passes) - 1
            n_wins_final = config.n_windows[final_pass_i]
            overlap_final = float(config.window_overlap[final_pass_i])
            ds_factor_final = int(config.ds_factor[final_pass_i])
            viz.export_velocity_profiles_pdf(
                vel_final=vel_final,
                interpolated_mask=interp_filled_mask,
                image=image_for_plots,
                n_windows=(int(n_wins_final[0]), int(n_wins_final[1])),
                overlap=overlap_final,
                ds_factor=ds_factor_final,
                scale_m_per_px=float(config.scale_m_per_px),
                flow_direction=str(config.flow_direction),
                time_s=time_s[: len(flow_ls)],
                output_path=plots_dir / "velocity_profiles.pdf",
            )

        if config.plot_flow_rate:
            print("Plotting: flow_rate.png")
            model_tuple: tuple[np.ndarray, np.ndarray] | None = None
            if config.plot_model:
                cough = CoughModel.from_gupta(
                    gender=config.model_gender,
                    weight_kg=float(config.model_mass),
                    height_m=float(config.model_height),
                )
                t_model, q_model = cough.flow(
                    time_s=time_s[: len(flow_ls)],
                    units="L/s",
                )
                model_tuple = (t_model, q_model)
            viz.plot_flow_rate(
                time_s[: len(flow_ls)],
                flow_ls,
                model=model_tuple,
                output_path=plots_dir / "flow_rate.png",
            )

    print(f"Done. Run directory: {run_dir}")
    return run_dir


def _relpath_or_name(path: str, *, base_dir: str) -> str:
    """Return a stable display path for logging/CSV outputs.

    We prefer relative paths (relative to the configured image directory) so outputs are
    portable. If that fails (e.g. different drives / unrelated paths), fall
    back to just the filename.
    """

    try:
        return str(Path(path).resolve().relative_to(Path(base_dir).resolve()))
    except Exception:
        return Path(path).name


def _apply_crop_and_background(imgs: np.ndarray, config: Config) -> np.ndarray:
    """Apply (optional) crop + background subtraction to 2D or 3D images."""

    out = imgs
    if any(config.crop_roi):
        print(f"Cropping images to ROI: {config.crop_roi}")
        out = crop(out, config.crop_roi)

    if config.background_dir:
        print(f"Background subtraction using: {config.background_dir}")
        bg = cv.imread(config.background_dir, cv.IMREAD_GRAYSCALE)
        if bg is None:
            raise RuntimeError(
                f"Failed to read background image: {config.background_dir}")

        if any(config.crop_roi):
            # Check if bg matches uncropped or cropped size
            h_in, w_in = imgs.shape[-2:]
            h_out, w_out = out.shape[-2:]
            h_bg, w_bg = bg.shape[:2]

            if (h_bg, w_bg) == (h_in, w_in):
                # Background matches original input -> apply crop
                bg = crop(bg, config.crop_roi)
            elif (h_bg, w_bg) == (h_out, w_out):
                # Background matches output (already cropped?) -> do not crop
                pass
            else:
                # Mismatch - try to guess based on width/height match
                # If width matches one but not the other, or neither match...
                # We can't safely proceed without knowing what to do.
                raise ValueError(
                    f"Background image shape {bg.shape} does not match input images {imgs.shape[-2:]} "
                    f"or cropped images {out.shape[-2:]}. Cannot safely subtract background."
                )

        out = np.clip(out.astype(np.int32) - bg.astype(np.int32),
                      0, None).astype(out.dtype)

    return out


def _neighbour_filter_strategy(
    pass_idx0: int,
    nb_thr: int | tuple[int, int],
) -> tuple[str, bool | str, str]:
    """Return (mode, replace, threshold_unit) for neighbour filtering."""

    if pass_idx0 == 0:
        return "xy", False, "std"
    if pass_idx0 == 1:
        return "r", True, "std"
    if isinstance(nb_thr, tuple):
        return "xy", "closest", "pxs"
    return "xy", True, "std"


if __name__ == "__main__":
    run()
