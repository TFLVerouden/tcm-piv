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
from typing import Any, cast

import cv2 as cv
import numpy as np

import tcm_piv as piv
from tcm_piv import init_config as cfg
from tcm_piv.preprocessing import crop
from tcm_piv.checkpoints import (
    init_run_dir,
    load_postprocessed_csv,
    load_unfiltered_peaks_csv_gz,
    pass_paths,
    write_meta_json,
    write_pairs_csv,
    write_postprocessed_csv,
    write_unfiltered_peaks_csv_gz,
    write_win_pos_csv,
)
from tcm_utils.io_utils import load_images
from tcm_utils.time_utils import timestamp_str


def run(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # CLI signature (positional):
    #   1) config_file (optional; if missing, a file picker prompt is used)
    #   2) resume_run_dir (optional; if provided, we resume into that folder)
    #
    # Example:
    #   python -m tcm_piv.run config.toml
    #   python -m tcm_piv.run config.toml output/runs/260110_004942

    config_file: Path | None
    if len(argv) >= 1 and argv[0]:
        config_file = Path(argv[0])
        argv = argv[1:]
    else:
        config_file = None

    # Optional: resume into an existing run directory.
    resume_run_dir: Path | None
    if len(argv) >= 1 and argv[0]:
        resume_run_dir = Path(argv[0])
    else:
        resume_run_dir = None

    cfg.read_file(config_file)

    # Each run gets its own directory so results are reproducible and
    # resumable. When resuming, the run folder is provided explicitly.
    run_id = timestamp_str()
    run_dir = resume_run_dir or init_run_dir(Path(cfg.OUTPUT_DIR), run_id)

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
    write_pairs_csv(run_dir / "pairs.csv", pairs)

    # Timebase used for plotting / any temporal processing.
    # `time_s` has length (NR_IMAGES - 1), matching the number of image pairs.
    frames = _frames_for_timebase()
    time_s = piv.get_time(frames, float(cfg.TIMESTEP_S))

    # Lazily loaded image stack. We only load if we need correlation.
    imgs: np.ndarray | None = None

    # Previous pass output (postprocessed) is used to compute shifts for the
    # next pass (multi-pass PIV refinement).
    prev_disp_final: np.ndarray | None = None

    for pass_i in range(cfg.NR_PASSES):
        pass_1b = pass_i + 1
        paths = pass_paths(run_dir, pass_1b)

        win_pos: np.ndarray | None = None

        n_wins = tuple(_per_pass(cfg.NR_WINDOWS, pass_i))
        n_wy, n_wx = int(n_wins[0]), int(n_wins[1])
        n_pairs = len(time_s)

        # Metadata is written once the pass completes.
        # It documents shapes, key config knobs, and the filenames that belong
        # to this pass. This is also the closest thing to a “checkpoint index”.
        meta: dict[str, Any] = {
            "pass": pass_1b,
            "n_pairs": n_pairs,
            "n_windows": [n_wy, n_wx],
            "downsample_factor": int(_per_pass(cfg.DOWNSAMPLE_FACTOR, pass_i)),
            "corrs_to_sum": int(_per_pass(cfg.CORRS_TO_SUM, pass_i)),
            "nr_peaks": int(_per_pass(cfg.NR_PEAKS, pass_i)),
            "min_peak_distance": int(_per_pass(cfg.MIN_PEAK_DISTANCE, pass_i)),
            "window_overlap": float(_per_pass(cfg.WINDOW_OVERLAP, pass_i)),
            "max_velocity_vx_vy_m_s": list(_per_pass(cfg.MAX_VELOCITY, pass_i)),
            "neighbourhood_size": list(_per_pass(cfg.NEIGHBOURHOOD_SIZE, pass_i)),
            "neighbourhood_threshold": _per_pass(cfg.NEIGHBOURHOOD_THRESHOLD, pass_i),
            "time_smoothing_lambda": float(_per_pass(cfg.TIME_SMOOTHING_LAMBDA, pass_i)),
            "files": {
                "pairs": "../pairs.csv",
                "unfiltered": paths.peaks_csv_gz.name,
                "post": paths.post_csv.name,
                "win_pos": paths.win_pos_csv.name,
            },
        }

        # --- Resume / checkpointing logic (file-based) ---
        #
        # Stage 2 (postprocessed) checkpoint is authoritative: if it exists,
        # the pass is considered complete. We still load it because later
        # passes need `prev_disp_final` to compute their shifts.
        if paths.post_csv.exists():
            _, _, _, disp_final = load_postprocessed_csv(
                paths.post_csv,
                n_pairs=n_pairs,
                n_wy=n_wy,
                n_wx=n_wx,
            )
            prev_disp_final = disp_final
            continue

        # Stage 1 checkpoint: correlation + peak finding already done.
        # This path is intentionally image-free (fast to resume).
        if paths.peaks_csv_gz.exists():
            disp_unf, int_unf = load_unfiltered_peaks_csv_gz(
                paths.peaks_csv_gz,
                n_pairs=n_pairs,
                n_wy=n_wy,
                n_wx=n_wx,
                n_peaks=int(_per_pass(cfg.NR_PEAKS, pass_i)),
            )
        else:
            # Full computation path: we need images to compute correlations.
            # We load once and reuse across passes.
            if imgs is None:
                imgs = _load_and_preprocess_images()

            # Later passes refine the search region by shifting windows based
            # on the previous pass result.
            if pass_i == 0:
                shifts = None
            else:
                if prev_disp_final is None:
                    raise RuntimeError(
                        f"Pass {pass_1b} needs previous displacement to compute shifts")
                shifts = piv.disp2shift((n_wy, n_wx), prev_disp_final)

            # 1) Calculate correlations per pair/window.
            corrs = piv.calc_corrs(
                imgs,
                n_wins=(n_wy, n_wx),
                shifts=shifts,
                overlap=float(_per_pass(cfg.WINDOW_OVERLAP, pass_i)),
                ds_fac=int(_per_pass(cfg.DOWNSAMPLE_FACTOR, pass_i)),
            )

            # 2) Optionally sum correlations over a time window.
            #    This is a denoising/smoothing step in correlation space.
            corrs_sum = piv.sum_corrs(
                corrs,
                int(_per_pass(cfg.CORRS_TO_SUM, pass_i)),
                n_wins=(n_wy, n_wx),
                shifts=shifts,
            )

            # 3) Find displacement peaks in the correlation planes.
            disp_unf, int_unf = piv.find_disps(
                corrs_sum,
                n_wins=(n_wy, n_wx),
                shifts=shifts,
                n_peaks=int(_per_pass(cfg.NR_PEAKS, pass_i)),
                ds_fac=int(_per_pass(cfg.DOWNSAMPLE_FACTOR, pass_i)),
                min_dist=int(_per_pass(cfg.MIN_PEAK_DISTANCE, pass_i)),
            )

            # Window positions (for plotting/interpretation) are derived from
            # the first image only (geometry only, no time dependence).
            _, win_pos = piv.split_n_shift(
                imgs[0],
                (n_wy, n_wx),
                overlap=float(_per_pass(cfg.WINDOW_OVERLAP, pass_i)),
            )

            # Persist stage-1 checkpoint.
            write_unfiltered_peaks_csv_gz(
                paths.peaks_csv_gz,
                disp_unf=disp_unf,
                int_unf=int_unf,
            )

        # Stage 2: postprocess the multi-peak results into a usable single-peak
        # displacement field, applying outlier and neighbour filtering.
        disp_glo, disp_nbs, disp_final = _postprocess_pass(
            pass_index_0b=pass_i,
            disp_unf=disp_unf,
            time_s=time_s,
        )

        # Persist stage-2 checkpoint.
        write_postprocessed_csv(
            paths.post_csv,
            time_s=time_s,
            disp_glo=disp_glo,
            disp_nbs=disp_nbs,
            disp_final=disp_final,
        )

        if win_pos is not None:
            write_win_pos_csv(paths.win_pos_csv, win_pos=win_pos)

        # Metadata is written last so it can be interpreted as
        # “this pass finished successfully and produced these artifacts”.
        write_meta_json(paths.meta_json, meta)
        prev_disp_final = disp_final

    print(f"Done. Run directory: {run_dir}")
    return 0


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


def _load_and_preprocess_images() -> np.ndarray:
    """Load all images and apply the preprocessing specified in config.

    Important: this can be expensive. `run()` deliberately delays calling
    this until it knows correlation work is required (i.e. no usable
    peak-detection checkpoint exists).
    """

    imgs = load_images(cfg.IMAGE_LIST, show_progress=True)
    imgs = np.asarray(imgs)

    if cfg.CROP_ROI != (0, 0, 0, 0):
        imgs = crop(imgs, cfg.CROP_ROI)

    if cfg.BACKGROUND_DIR:
        bg = cv.imread(cfg.BACKGROUND_DIR, cv.IMREAD_GRAYSCALE)
        if bg is None:
            raise RuntimeError(
                f"Failed to load background image: {cfg.BACKGROUND_DIR}")
        if cfg.CROP_ROI != (0, 0, 0, 0):
            bg = crop(bg, cfg.CROP_ROI)

        imgs_i32 = imgs.astype(np.int32)
        bg_i32 = bg.astype(np.int32)
        imgs = np.clip(imgs_i32 - bg_i32, 0, None).astype(imgs.dtype)

    return imgs


def _frames_for_timebase() -> list[int]:
    """Determine the frame indices used to compute time stamps.

    `cfg.FRAMES_TO_USE` is either:
    - a list of integers, or
    - the string "all", which is interpreted as 1..NR_IMAGES (inclusive)

    These indices are passed to `piv.get_time()` to produce `time_s`.
    """

    if isinstance(cfg.FRAMES_TO_USE, list):
        return cfg.FRAMES_TO_USE
    # When "all" is used, we typically deal with 1-based frame naming.
    return list(range(1, cfg.NR_IMAGES + 1))


def _dmax_px(*, vx_max_m_s: float, vy_max_m_s: float) -> tuple[float, float]:
    """Convert velocity limits from m/s into displacement limits in pixels.

    Postprocessing outlier filtering works in pixel-displacement space.
    Given a timestep (s) and a spatial scale (m/px), the maximum displacement
    in pixels is computed as:
    `d_px = v_m_s * timestep_s / scale_m_per_px`
    """

    if not cfg.SCALE_M_PER_PX:
        raise RuntimeError("Missing SCALE_M_PER_PX from calibration metadata")
    if not cfg.TIMESTEP_S:
        raise RuntimeError("Missing TIMESTEP_S from camera metadata")
    a_y = vy_max_m_s * cfg.TIMESTEP_S / cfg.SCALE_M_PER_PX
    b_x = vx_max_m_s * cfg.TIMESTEP_S / cfg.SCALE_M_PER_PX
    return float(a_y), float(b_x)


def _odd_at_most(value: int, max_value: int) -> int:
    """Clamp to [1, max_value] and force the result to be odd.

    Neighbourhood filters often assume an odd window size so there is a
    well-defined center element.
    """

    v = int(value)
    m = int(max_value)
    if m <= 1:
        return 1
    v = min(max(v, 1), m)
    if v % 2 == 0:
        v -= 1
    return max(v, 1)


def _clamp_n_nbs(n_nbs: tuple[int, int, int], *, shape_3: tuple[int, int, int]) -> tuple[int, int, int]:
    """Clamp neighbourhood sizes to the available (corrs, win_y, win_x) shape."""

    n_corrs, n_wy, n_wx = shape_3
    return (
        _odd_at_most(n_nbs[0], n_corrs),
        _odd_at_most(n_nbs[1], n_wy),
        _odd_at_most(n_nbs[2], n_wx),
    )


def _postprocess_pass(
    *,
    pass_index_0b: int,
    disp_unf: np.ndarray,
    time_s: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray]:
    """Return (disp_glo, disp_nbs, disp_final) in 4D form.

    Notes:
    - disp_unf is 5D (pairs, wy, wx, peaks, 2).
    - Some replacement modes ("closest") require keeping multiple peaks until
      after neighbour filtering.
    """

    # max_velocity config is (vx_max, vy_max) in m/s.
    # We convert it to pixel displacement limits for global outlier filtering.
    vx_max, vy_max = _per_pass(cfg.MAX_VELOCITY, pass_index_0b)
    a_y, b_x = _dmax_px(vx_max_m_s=float(vx_max), vy_max_m_s=float(vy_max))

    thr = _per_pass(cfg.NEIGHBOURHOOD_THRESHOLD, pass_index_0b)
    n_nbs = _per_pass(cfg.NEIGHBOURHOOD_SIZE, pass_index_0b)
    lam = float(_per_pass(cfg.TIME_SMOOTHING_LAMBDA, pass_index_0b))

    # Neighbour filtering behavior differs by pass, to mirror the legacy
    # workflow in `piv/piv.py`. These knobs are intentionally *not* exposed in
    # the TOML yet; if you want them configurable, we can add config keys.
    if pass_index_0b == 0:
        nb_mode: str = "xy"
        nb_replace: bool | str = False
        nb_thr_unit = "std"
    elif pass_index_0b == 1:
        nb_mode = "r"
        nb_replace = True  # median replacement
        nb_thr_unit = "std"
    else:
        nb_mode = "xy"
        if isinstance(thr, tuple):
            nb_replace = "closest"
            nb_thr_unit = "pxs"
        else:
            nb_replace = True
            nb_thr_unit = "std"

    # Global outlier filter is applied *before* selecting a single peak.
    # At this stage we still carry the extra "peaks" dimension.
    disp_glo_5d = np.asarray(
        piv.filter_outliers(
            "semicircle_rect",
            disp_unf,
            a=a_y,
            b=b_x,
            verbose=True,
        )
    )

    # Safety: clamp neighbourhood sizes to the available data dimensions.
    # The neighbourhood size is (n_corrs, n_wy, n_wx) and must be odd.
    # Make shapes explicit for type-checkers.
    disp_shape_3: tuple[int, int, int] = (
        int(disp_glo_5d.shape[0]),
        int(disp_glo_5d.shape[1]),
        int(disp_glo_5d.shape[2]),
    )
    n_nbs_t = tuple(int(x) for x in n_nbs)
    if len(n_nbs_t) != 3:
        raise ValueError(
            f"neighbourhood_size must have 3 elements, got {n_nbs_t}")
    n_nbs = _clamp_n_nbs(
        cast(tuple[int, int, int], n_nbs_t), shape_3=disp_shape_3)

    if nb_replace == "closest":
        # Keep multiple peaks for neighbour filter; strip afterwards.
        disp_nbs_5d = piv.filter_neighbours(
            disp_glo_5d,
            thr=thr,  # type: ignore[arg-type]
            thr_unit=nb_thr_unit,
            n_nbs=n_nbs,
            mode=nb_mode,
            replace=nb_replace,
            verbose=True,
            timing=True,
        )

        disp_glo = piv.strip_peaks(
            disp_glo_5d, axis=-2, mode="reduce", verbose=False)
        disp_nbs = piv.strip_peaks(
            disp_nbs_5d, axis=-2, mode="reduce", verbose=True)
        disp_final = disp_nbs
    else:
        disp_glo = piv.strip_peaks(
            disp_glo_5d, axis=-2, mode="reduce", verbose=True)
        disp_nbs = piv.filter_neighbours(
            disp_glo,
            thr=thr,  # type: ignore[arg-type]
            n_nbs=n_nbs,
            mode=nb_mode,
            replace=nb_replace,
            verbose=True,
            timing=True,
        )
        disp_final = disp_nbs

    # Optional temporal smoothing.
    # This is only meaningful for time series at a single window location
    # (n_wy, n_wx) == (1, 1). For multi-window fields, smoothing would need a
    # different approach (or be applied per-window).
    if lam and lam > 0:
        # `smooth()` expects something that can be squeezed to (n_time, 2).
        # This is typically pass 1 with a single window and many time points.
        if disp_final.shape[0] >= 3 and disp_final.shape[1:3] == (1, 1):
            disp_final = piv.smooth(time_s, disp_final, lam=lam, type=int)

    return disp_glo, disp_nbs, disp_final


if __name__ == "__main__":
    raise SystemExit(run())
