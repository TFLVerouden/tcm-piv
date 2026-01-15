"""CLI entrypoint for tcm-piv.

This file orchestrates the end-to-end PIV pipeline:
- Read/normalize a TOML config via :mod:`tcm_piv.init_config` (aliased as `cfg`).
- Create or reuse a run directory under `<output_dir>/runs/<run_id>/`.
- For each pass, produce two *checkpoints* on disk:

    1. **Unfiltered peak-detection results** (`pass_XX_unfiltered.csv.gz`)
         - Created right after correlation + peak finding.
         - Contains *all* peaks (multi-peak), so it can be postprocessed later
             without needing to re-load images.

    2. **Postprocessed results** (`pass_XX_post.csv`)
         - Created after global filtering + neighbour filtering (+ optional
             temporal smoothing).
         - Contains the “final” single-peak displacement field plus some
             intermediate columns.

In addition to the checkpoints, each pass writes `pass_XX_meta.json` (small)
and optionally `pass_XX_win_pos.csv` (window center positions for plotting).
The mapping from `pair_index -> image0,image1` is written once in `pairs.csv`
in the run root.

---

How to run
----------

This module is designed to be run as a module:

`python -m tcm_piv.run path/to/config.toml`

The config controls both the *input* (image directory + camera/calibration
metadata) and the *output directory* where runs are created.

How to resume / continue
------------------------

`run()` supports resuming into an existing run folder by passing it as the
second positional argument:

`python -m tcm_piv.run path/to/config.toml path/to/output/runs/<run_id>`

Resume logic is purely file-based:
- If `pass_XX_post.csv` exists, that pass is considered “done” and will be
    skipped (the file is loaded to provide `prev_disp_final` for the next pass).
- Else if `pass_XX_unfiltered.csv.gz` exists, correlation/peak detection is
    skipped and only postprocessing is run.
- Else the full pass is computed from images.

Resuming from a *specific* checkpoint stage
------------------------------------------

There is no single “checkpoint list” file; the run folder *is* the list.
Each pass folder contains the canonical filenames and a `pass_XX_meta.json`
that also records them.

If you want to restart at a specific stage, the simplest approach is to
delete the later-stage files and re-run with the same run folder:

- Re-run *postprocessing only* for a pass:
    - Keep `pass_XX_unfiltered.csv.gz`
    - Delete `pass_XX_post.csv`

- Re-run *correlation + peak detection + postprocessing* for a pass:
    - Delete `pass_XX_unfiltered.csv.gz` and `pass_XX_post.csv`

Because image loading is only needed for correlation, the “postprocessing-only”
resume path is typically much faster and avoids re-reading image data.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import cv2 as cv
import numpy as np

import tcm_piv as piv
import tcm_piv.visualisation as viz
from tcm_piv import init_config as cfg
from tcm_piv.preprocessing import crop
from tcm_utils.cough_model import CoughModel
from tcm_piv.checkpoints import (
    init_run_dir,
    load_postprocessed_csv,
    load_unfiltered_peaks_csv_gz,
    pass_paths,
    write_meta_json,
    write_pairs_csv,
    write_postprocessed_csv,
    write_unfiltered_peaks_csv_gz,
)
from tcm_utils.io_utils import load_images
from tcm_utils.time_utils import timestamp_str


def run(
    *,
    config_file: str | Path | None = None,
    resume_run_dir: str | Path | None = None,
    start_pass_1b: int = 1,
) -> Path:
    """Run the PIV pipeline.

    Args:
        config_file: Path to a TOML config file. If None, a file picker prompt is used.
        resume_run_dir: Existing run directory to resume into. If None, a new run is created.
        start_pass_1b: 1-based pass index to start at when resuming.

    Returns:
        run_dir (Path): Path to the run directory where results are stored.
    """

    print("\n\nStarting PIV analysis...")

    if start_pass_1b < 1:
        raise ValueError("start_pass_1b must be >= 1")

    config_path = Path(config_file) if config_file else None
    resume_path = Path(resume_run_dir) if resume_run_dir else None

    if config_path is not None:
        print(f"Config file: {config_path}")
    if resume_path is not None:
        print(f"Resume run dir: {resume_path}")
    if start_pass_1b != 1:
        print(f"Start pass (1-based): {start_pass_1b}")

    print("\nReading config...")
    cfg.read_file(config_path)

    print("Config summary:")
    print(f"  image_dir: {cfg.IMAGE_DIR}")
    print(f"  nr_images: {cfg.NR_IMAGES}")
    print(f"  nr_passes: {cfg.NR_PASSES}")
    print(f"  timestep_s: {cfg.TIMESTEP_S}")
    print(f"  scale_m_per_px: {cfg.SCALE_M_PER_PX}")
    print(f"  output_dir: {cfg.OUTPUT_DIR}")

    # Each run gets its own directory so results are reproducible and
    # resumable. When resuming, the run folder is provided explicitly.
    run_id = timestamp_str()
    run_dir = resume_path or init_run_dir(Path(cfg.OUTPUT_DIR), run_id)
    print(f"\nRun directory: {run_dir}")

    # Pair mapping for CSV row -> image filenames.
    #
    # All result CSVs refer to images by *pair_index* (0-based), not by
    # embedding filenames in every row. `pairs.csv` is the lookup table.
    # This keeps large CSV outputs compact.
    pairs = [
        (
            i,
            _relpath_or_name(cfg.IMAGE_LIST[i], base_dir=cfg.IMAGE_DIR),
            _relpath_or_name(cfg.IMAGE_LIST[i + 1], base_dir=cfg.IMAGE_DIR),
        )
        for i in range(cfg.NR_IMAGES - 1)
    ]
    print(f"Writing pairs.csv ({len(pairs)} pairs)...")
    write_pairs_csv(run_dir / "pairs.csv", pairs)

    # Timebase used for plotting / any temporal processing.
    # `time_s` has length (NR_IMAGES - 1), matching the number of image pairs.
    if isinstance(cfg.FRAMES_TO_USE, list):
        frames = cfg.FRAMES_TO_USE
    else:
        frames = list(range(1, cfg.NR_IMAGES + 1))
    print(f"Timebase: {len(frames)} frames -> {len(frames) - 1} pairs")
    time_s = piv.get_time(frames, float(cfg.TIMESTEP_S))

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

    if start_pass_1b > cfg.NR_PASSES:
        raise ValueError(
            f"start_pass ({start_pass_1b}) exceeds configured nr_passes ({cfg.NR_PASSES})"
        )

    for pass_idx0 in range(cfg.NR_PASSES):
        pass_idx1 = pass_idx0 + 1
        paths = pass_paths(run_dir, pass_idx1)

        plot_windows = bool(_per_pass(cfg.WINDOW_PLOT_ENABLED, pass_idx0))
        # vector overlays removed for now

        n_wins = tuple(_per_pass(cfg.NR_WINDOWS, pass_idx0))
        n_win_y, n_win_x = int(n_wins[0]), int(n_wins[1])
        n_img_pairs = len(time_s)

        print(
            f"\nPASS {pass_idx1:02d}/{int(cfg.NR_PASSES):02d}: n_wins=({n_win_y},{n_win_x})"
        )

        if pass_idx1 < start_pass_1b:
            if resume_path is None:
                raise ValueError(
                    "start_pass_1b > 1 requires resume_run_dir pointing to an existing run"
                )
            if not paths.post_csv.exists():
                raise RuntimeError(
                    f"Requested start at pass {start_pass_1b}, but prior pass {pass_idx1} has no post file: {paths.post_csv}"
                )
            print(
                f"Skipping pass {pass_idx1:02d} (before start_pass). Loading post checkpoint: {paths.post_csv.name}"
            )
            _, _, _, prev_disp_final = load_postprocessed_csv(
                paths.post_csv,
                n_pairs=n_img_pairs,
                n_wy=n_win_y,
                n_wx=n_win_x,
            )
            continue

        # Config ordering is always [vy, vx].
        vy_max_m_s, vx_max_m_s = _per_pass(cfg.MAX_VELOCITY, pass_idx0)
        filter_ranges_vx_vy.append((float(vx_max_m_s), float(vy_max_m_s)))

        nb_thr = _per_pass(cfg.NEIGHBOURHOOD_THRESHOLD, pass_idx0)
        nb_size_tyx = tuple(int(x)
                            for x in _per_pass(cfg.NEIGHBOURHOOD_SIZE, pass_idx0))
        interp_nb_cfg = _per_pass(cfg.INTERPOLATION_NEIGHBOURHOOD, pass_idx0)
        interp_nb_size_tyx: tuple[int, int, int] | None
        if interp_nb_cfg is None:
            interp_nb_size_tyx = None
        else:
            interp_nb_size_tyx = (
                int(interp_nb_cfg[0]),
                int(interp_nb_cfg[1]),
                int(interp_nb_cfg[2]),
            )
        time_smooth_lam = float(
            _per_pass(cfg.TIME_SMOOTHING_LAMBDA, pass_idx0))

        ds_factor = int(_per_pass(cfg.DOWNSAMPLE_FACTOR, pass_idx0))
        n_corrs_to_sum = int(_per_pass(cfg.CORRS_TO_SUM, pass_idx0))
        n_peaks = int(_per_pass(cfg.NR_PEAKS, pass_idx0))
        min_peak_dist_px = int(_per_pass(cfg.MIN_PEAK_DISTANCE, pass_idx0))
        overlap = float(_per_pass(cfg.WINDOW_OVERLAP, pass_idx0))

        print("Pass parameters:")
        print(f"  downsample_factor: {ds_factor}")
        print(f"  corrs_to_sum: {n_corrs_to_sum}")
        print(f"  nr_peaks: {n_peaks}")
        print(f"  min_peak_distance_px: {min_peak_dist_px}")
        print(f"  window_overlap: {overlap}")
        print(f"  max_velocity (vy,vx) [m/s]: ({vy_max_m_s},{vx_max_m_s})")

        # Metadata is written once the pass completes.
        # It documents shapes, key config knobs, and the filenames that belong
        # to this pass. This is also the closest thing to a “checkpoint index”.
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
            "outlier_filter_mode": str(_per_pass(cfg.OUTLIER_FILTER_MODE, pass_idx0)).strip().lower(),
            "neighbourhood_size": list(nb_size_tyx),
            "neighbourhood_threshold": nb_thr,
            "time_smoothing_lambda": time_smooth_lam,
            "files": {
                "pairs": "../pairs.csv",
                "unfiltered": paths.peaks_csv_gz.name,
                "post": paths.post_csv.name,
            },
        }

        # --- Resume / checkpointing logic (file-based) ---
        #
        # Stage 2 (postprocessed) checkpoint is authoritative: if it exists,
        # the pass is considered complete. We still load it because later
        # passes need `prev_disp_final` to compute their shifts.
        if paths.post_csv.exists():
            print(
                f"Found postprocessed checkpoint: {paths.post_csv.name} -> skipping correlation + postprocessing"
            )
            _, _, _, disp_final = load_postprocessed_csv(
                paths.post_csv,
                n_pairs=n_img_pairs,
                n_wy=n_win_y,
                n_wx=n_win_x,
            )
            if plot_windows:
                shifts = None
                if pass_idx0 > 0 and prev_disp_final is not None:
                    shifts = piv.disp2shift(
                        (n_win_y, n_win_x), prev_disp_final)
                window_layouts.append({
                    "pass": pass_idx1,
                    "n_wins": (n_win_y, n_win_x),
                    "overlap": overlap,
                    "shifts": shifts,
                })

            if pass_idx0 == cfg.NR_PASSES - 1:
                disp_final_lastpass = disp_final
                # Best-effort load of interpolation mask (if previously saved).
                mask_csv = run_dir / "interpolated_mask.csv"
                if mask_csv.exists():
                    try:
                        mask_rows = np.loadtxt(
                            mask_csv, delimiter=",", skiprows=1)
                        if mask_rows.ndim == 1 and mask_rows.size == 0:
                            interp_filled_mask = None
                        else:
                            mask_flat = mask_rows[:, 3].astype(bool)
                            interp_filled_mask = mask_flat.reshape(
                                n_img_pairs, n_win_y, n_win_x)
                    except Exception:
                        interp_filled_mask = None
            prev_disp_final = disp_final
            continue

        # Stage 1 checkpoint: correlation + peak finding already done.
        # This path is intentionally image-free (fast to resume).
        if paths.peaks_csv_gz.exists():
            print(
                f"Found unfiltered checkpoint: {paths.peaks_csv_gz.name} -> skipping correlation + peak finding"
            )
            disp_peaks_unf, peak_int_unf = load_unfiltered_peaks_csv_gz(
                paths.peaks_csv_gz,
                n_pairs=n_img_pairs,
                n_wy=n_win_y,
                n_wx=n_win_x,
                n_peaks=n_peaks,
            )
        else:
            # Full computation path: we need images to compute correlations.
            # We load once and reuse across passes.
            if imgs is None:
                print("Loading images...")
                imgs = load_images(cfg.IMAGE_LIST, show_progress=True)
                imgs = np.asarray(imgs)

                if cfg.CROP_ROI != (0, 0, 0, 0):
                    print(f"Cropping images to ROI: {cfg.CROP_ROI}")
                    imgs = crop(imgs, cfg.CROP_ROI)

                if cfg.BACKGROUND_DIR:
                    print(
                        f"Background subtraction using: {cfg.BACKGROUND_DIR}")
                    bg = cv.imread(cfg.BACKGROUND_DIR, cv.IMREAD_GRAYSCALE)
                    if bg is not None:
                        if cfg.CROP_ROI != (0, 0, 0, 0):
                            bg = crop(bg, cfg.CROP_ROI)
                        imgs = np.clip(
                            imgs.astype(np.int32) -
                            bg.astype(np.int32), 0, None
                        ).astype(imgs.dtype)

            # Later passes refine the search region by shifting windows based
            # on the previous pass result.
            if pass_idx0 == 0:
                shifts = None
            else:
                if prev_disp_final is None:
                    raise RuntimeError(
                        f"Pass {pass_idx1} needs previous displacement to compute shifts")
                print("Computing window shifts from previous pass...")
                shifts = piv.disp2shift((n_win_y, n_win_x), prev_disp_final)

            # 1) Calculate correlations per pair/window.
            print("Step 1: calculating correlation maps...")
            corrs = piv.calc_corrs(
                imgs,
                n_wins=(n_win_y, n_win_x),
                shifts=shifts,
                overlap=overlap,
                ds_fac=ds_factor,
            )

            # 2) Optionally sum correlations over a time window.
            #    This is a denoising/smoothing step in correlation space.
            print(
                f"Step 2: summing correlation maps (corrs_to_sum={n_corrs_to_sum})...")
            corrs_sum = piv.sum_corrs(
                corrs,
                n_corrs_to_sum,
                n_wins=(n_win_y, n_win_x),
                shifts=shifts,
            )

            if bool(getattr(cfg, "PLOT_CORRELATIONS", False)):
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
                n_wins=(n_win_y, n_win_x),
                shifts=shifts,
                n_peaks=n_peaks,
                ds_fac=ds_factor,
                min_dist=min_peak_dist_px,
                subpx=pass_idx0 == cfg.NR_PASSES - 1,
            )

            # Window positions (for plotting/interpretation) are derived from
            # the first image only (geometry only, no time dependence).
            # Persist stage-1 checkpoint.
            print(f"Writing unfiltered checkpoint: {paths.peaks_csv_gz.name}")
            write_unfiltered_peaks_csv_gz(
                paths.peaks_csv_gz,
                disp_unf=disp_peaks_unf,
                int_unf=peak_int_unf,
            )

        # Stage 2: postprocess the multi-peak results into a usable single-peak
        # displacement field, applying outlier and neighbour filtering.
        print("Postprocessing: outlier + neighbour filtering")
        a_y = float(vy_max_m_s) * float(cfg.TIMESTEP_S) / \
            float(cfg.SCALE_M_PER_PX)
        b_x = float(vx_max_m_s) * float(cfg.TIMESTEP_S) / \
            float(cfg.SCALE_M_PER_PX)
        a_x = float(vx_max_m_s) * float(cfg.TIMESTEP_S) / \
            float(cfg.SCALE_M_PER_PX)
        b_y = float(vy_max_m_s) * float(cfg.TIMESTEP_S) / \
            float(cfg.SCALE_M_PER_PX)

        if pass_idx0 == 0:
            nb_mode: str = "xy"
            nb_replace: bool | str = False
            nb_thr_unit = "std"
        elif pass_idx0 == 1:
            nb_mode = "r"
            nb_replace = True
            nb_thr_unit = "std"
        else:
            nb_mode = "xy"
            if isinstance(nb_thr, tuple):
                nb_replace = "closest"
                nb_thr_unit = "pxs"
            else:
                nb_replace = True
                nb_thr_unit = "std"

        outlier_mode = str(
            _per_pass(cfg.OUTLIER_FILTER_MODE, pass_idx0)).strip().lower()
        flow_dir = str(cfg.FLOW_DIRECTION).strip().lower()
        if flow_dir not in {"x", "y"}:
            raise ValueError("cfg.FLOW_DIRECTION must be 'x' or 'y'")

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
            a_px = vmax * float(cfg.TIMESTEP_S) / float(cfg.SCALE_M_PER_PX)
            b_px = None
            disp_for_filter = disp_peaks_unf
            unswap = False
        else:
            raise ValueError(
                "cfg.OUTLIER_FILTER_MODE must be 'semicircle_rect' or 'circle'"
            )

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
                f"neighbourhood_size must have 3 elements, got {nb_size_tyx}")

        if interp_nb_size_tyx is not None and len(interp_nb_size_tyx) != 3:
            raise ValueError(
                f"interpolation_neighbourhood must have 3 elements, got {interp_nb_size_tyx}")

        if nb_replace == "closest":
            print(
                f"  neighbour_filter: mode={nb_mode} thr_unit={nb_thr_unit} replace={nb_replace}"
            )
            disp_nb_5d = piv.filter_neighbours(
                disp_global_5d,
                thr=nb_thr,  # type: ignore[arg-type]
                thr_unit=nb_thr_unit,
                n_nbs=nb_size_tyx,
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
                f"  neighbour_filter: mode={nb_mode} thr_unit={nb_thr_unit} replace={nb_replace}"
            )
            disp_global = piv.strip_peaks(
                disp_global_5d, axis=-2, mode="reduce", verbose=True
            )
            disp_nb = piv.filter_neighbours(
                disp_global,
                thr=nb_thr,  # type: ignore[arg-type]
                n_nbs=nb_size_tyx,
                mode=nb_mode,
                replace=nb_replace,
                verbose=True,
                timing=True,
            )
            disp_final = disp_nb

        if time_smooth_lam and time_smooth_lam > 0:
            if disp_final.shape[0] >= 3 and disp_final.shape[1:3] == (1, 1):
                print(
                    f"  temporal_smoothing: enabled (lambda={time_smooth_lam})"
                )
                disp_final = piv.smooth(
                    time_s, disp_final, lam=time_smooth_lam, type=int)

        # Final pass: patch remaining NaN holes by interpolation
        if (
            pass_idx0 == cfg.NR_PASSES - 1
            and interp_nb_size_tyx is not None
            and any(v > 1 for v in interp_nb_size_tyx)
            and np.isnan(disp_final).any()
        ):
            print(
                f"  interpolation: patching NaN holes (n_nbs={interp_nb_size_tyx})"
            )
            disp_pre_interp = disp_final
            nan_mask = np.any(np.isnan(disp_pre_interp), axis=-1)

            disp_patched = piv.filter_neighbours(
                disp_pre_interp,
                thr=None,
                n_nbs=interp_nb_size_tyx,
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
                "n_wins": (n_win_y, n_win_x),
                "overlap": overlap,
                "shifts": shifts,
            })

        if pass_idx0 == cfg.NR_PASSES - 1:
            disp_final_lastpass = disp_final

        # Persist stage-2 checkpoint.
        print(f"Writing postprocessed checkpoint: {paths.post_csv.name}")
        write_postprocessed_csv(
            paths.post_csv,
            time_s=time_s,
            disp_glo=disp_global,
            disp_nbs=disp_nb,
            disp_final=disp_final,
        )

        # Window positions are no longer persisted; plotting uses `split_n_shift(plot=True)`.

        # Metadata is written last so it can be interpreted as
        # “this pass finished successfully and produced these artifacts”.
        print(f"Writing pass metadata: {paths.meta_json.name}")
        write_meta_json(paths.meta_json, meta)
        prev_disp_final = disp_final

    if disp_final_lastpass is not None:
        print("\nFinal exports: velocity + flow rate")
        vel_final = disp_final_lastpass * \
            float(cfg.SCALE_M_PER_PX) / float(cfg.TIMESTEP_S)
        flow_m3s = piv.vel2flow(
            vel_final,
            float(cfg.EXTRA_VEL_DIM_M),
            float(cfg.IMAGE_WIDTH_M),
            float(cfg.IMAGE_HEIGHT_M),
            flow_direction=str(cfg.FLOW_DIRECTION),
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
        if cfg.PLOT_GLOBAL_FILTERS and filter_ranges_vx_vy:
            print("Plotting: filter_ranges.png")
            viz.plot_filter_ranges(
                filter_ranges_vx_vy,
                mode=cfg.OUTLIER_FILTER_MODE,
                flow_direction=str(cfg.FLOW_DIRECTION),
                output_path=plots_dir / "filter_ranges.png",
            )

        image_for_plots: np.ndarray | None = None
        if window_layouts or cfg.EXPORT_VELOCITY_PROFILES_PDF:
            if imgs is not None:
                image_for_plots = np.asarray(imgs[0])
            else:
                base = load_images([cfg.IMAGE_LIST[0]], show_progress=False)
                image_for_plots = np.asarray(base[0])
                if cfg.CROP_ROI != (0, 0, 0, 0):
                    image_for_plots = crop(image_for_plots, cfg.CROP_ROI)
                if cfg.BACKGROUND_DIR:
                    bg = cv.imread(cfg.BACKGROUND_DIR, cv.IMREAD_GRAYSCALE)
                    if bg is not None:
                        if cfg.CROP_ROI != (0, 0, 0, 0):
                            bg = crop(bg, cfg.CROP_ROI)
                        image_for_plots = np.clip(
                            image_for_plots.astype(np.int32) -
                            bg.astype(np.int32), 0, None
                        ).astype(image_for_plots.dtype)

        if window_layouts and image_for_plots is not None:
            print("Plotting: window layouts")
            for layout in window_layouts:
                viz.plot_window_layout(
                    image_for_plots,
                    layout["n_wins"],
                    overlap=float(layout.get("overlap", 0.0)),
                    shifts=layout.get("shifts"),
                    shift_mode="before",
                    title=f"Pass {layout['pass']:02d} windows",
                    output_path=plots_dir /
                    f"pass_{layout['pass']:02d}_windows.png",
                )

        if cfg.EXPORT_VELOCITY_PROFILES_PDF and image_for_plots is not None:
            print("Exporting: velocity_profiles.pdf")
            final_pass_i = int(cfg.NR_PASSES) - 1
            n_wins_final = tuple(_per_pass(cfg.NR_WINDOWS, final_pass_i))
            overlap_final = float(_per_pass(cfg.WINDOW_OVERLAP, final_pass_i))
            ds_fac_final = int(_per_pass(cfg.DOWNSAMPLE_FACTOR, final_pass_i))
            viz.export_velocity_profiles_pdf(
                vel_final=vel_final,
                interpolated_mask=interp_filled_mask,
                image=image_for_plots,
                n_wins=(int(n_wins_final[0]), int(n_wins_final[1])),
                overlap=overlap_final,
                ds_fac=ds_fac_final,
                scale_m_per_px=float(cfg.SCALE_M_PER_PX),
                flow_direction=str(cfg.FLOW_DIRECTION),
                time_s=time_s[: len(flow_ls)],
                output_path=plots_dir / "velocity_profiles.pdf",
            )

        if cfg.PLOT_FLOW_RATE:
            print("Plotting: flow_rate.png")
            model_tuple: tuple[np.ndarray, np.ndarray] | None = None
            if cfg.PLOT_MODEL:
                cough = CoughModel.from_gupta(
                    gender=cfg.MODEL_GENDER,
                    weight_kg=float(cfg.MODEL_MASS),
                    height_m=float(cfg.MODEL_HEIGHT),
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


def _per_pass(value: Any, idx: int) -> Any:
    """Return per-pass element, supporting either scalar or list."""

    return value[idx] if isinstance(value, list) else value


def _relpath_or_name(path: str, *, base_dir: str) -> str:
    """Return a stable display path for logging/CSV outputs.

    We prefer relative paths (relative to `cfg.IMAGE_DIR`) so outputs are
    portable. If that fails (e.g. different drives / unrelated paths), fall
    back to just the filename.
    """

    try:
        return str(Path(path).resolve().relative_to(Path(base_dir).resolve()))
    except Exception:
        return Path(path).name


if __name__ == "__main__":
    # Local/dev convenience: hardcode a config for quick manual testing.
    # Keep `run()` itself as a callable API.
    run_dir = run()
    print(run_dir)
    raise SystemExit(0)
