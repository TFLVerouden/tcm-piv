"""CLI entrypoint for tcm-piv.

This script is a structured port of the legacy workflow in piv/piv.py:
- Run multiple passes (nr_passes)
- For each pass:
  1) correlate + peak detection -> checkpoint (CSV.gz)
  2) postprocess -> checkpoint (CSV)

The goal is to keep main() lean while reusing the existing tcm_piv submodule
functions (correlation/displacement/postprocessing/plotting).
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

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


def _per_pass(value: Any, idx: int) -> Any:
    """Return per-pass element, supporting either scalar or list."""

    return value[idx] if isinstance(value, list) else value


def _relpath_or_name(path: str, *, base_dir: str) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(base_dir).resolve()))
    except Exception:
        return Path(path).name


def _load_and_preprocess_images() -> np.ndarray:
    imgs = load_images(cfg.IMAGE_LIST, show_progress=True)
    imgs = np.asarray(imgs)

    if cfg.CROP_ROI != (0, 0, 0, 0):
        imgs = crop(imgs, cfg.CROP_ROI)

    if cfg.BACKGROUND_DIR:
        bg = cv.imread(cfg.BACKGROUND_DIR, cv.IMREAD_GRAYSCALE)
        if bg is None:
            raise RuntimeError(f"Failed to load background image: {cfg.BACKGROUND_DIR}")
        if cfg.CROP_ROI != (0, 0, 0, 0):
            bg = crop(bg, cfg.CROP_ROI)

        imgs_i32 = imgs.astype(np.int32)
        bg_i32 = bg.astype(np.int32)
        imgs = np.clip(imgs_i32 - bg_i32, 0, None).astype(imgs.dtype)

    return imgs


def _frames_for_timebase() -> list[int]:
    if isinstance(cfg.FRAMES_TO_USE, list):
        return cfg.FRAMES_TO_USE
    # When "all" is used, we typically deal with 1-based frame naming.
    return list(range(1, cfg.NR_IMAGES + 1))


def _dmax_px(*, vx_max_m_s: float, vy_max_m_s: float) -> tuple[float, float]:
    if not cfg.SCALE_M_PER_PX:
        raise RuntimeError("Missing SCALE_M_PER_PX from calibration metadata")
    if not cfg.TIMESTEP_S:
        raise RuntimeError("Missing TIMESTEP_S from camera metadata")
    a_y = vy_max_m_s * cfg.TIMESTEP_S / cfg.SCALE_M_PER_PX
    b_x = vx_max_m_s * cfg.TIMESTEP_S / cfg.SCALE_M_PER_PX
    return float(a_y), float(b_x)


def _odd_at_most(value: int, max_value: int) -> int:
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

    # max_velocity config is (vx_max, vy_max)
    vx_max, vy_max = _per_pass(cfg.MAX_VELOCITY, pass_index_0b)
    a_y, b_x = _dmax_px(vx_max_m_s=float(vx_max), vy_max_m_s=float(vy_max))

    thr = _per_pass(cfg.NEIGHBOURHOOD_THRESHOLD, pass_index_0b)
    n_nbs = _per_pass(cfg.NEIGHBOURHOOD_SIZE, pass_index_0b)
    lam = float(_per_pass(cfg.TIME_SMOOTHING_LAMBDA, pass_index_0b))

    # Mirror piv.py defaults for the existing 3-pass workflow.
    # (If you later want these configurable, we can add config keys.)
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

    # Global filter (still multi-peak at this point)
    disp_glo_5d = np.asarray(
        piv.filter_outliers(
        "semicircle_rect",
        disp_unf,
        a=a_y,
        b=b_x,
        verbose=True,
        )
    )

    n_nbs = _clamp_n_nbs(tuple(int(x) for x in n_nbs), shape_3=disp_glo_5d.shape[:3])

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

        disp_glo = piv.strip_peaks(disp_glo_5d, axis=-2, mode="reduce", verbose=False)
        disp_nbs = piv.strip_peaks(disp_nbs_5d, axis=-2, mode="reduce", verbose=True)
        disp_final = disp_nbs
    else:
        disp_glo = piv.strip_peaks(disp_glo_5d, axis=-2, mode="reduce", verbose=True)
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

    # Smooth for next pass (only meaningful for (1, 1) windows)
    if lam and lam > 0:
        # `smooth()` expects something that can be squeezed to (n_time, 2).
        # This is typically pass 1 with a single window and many time points.
        if disp_final.shape[0] >= 3 and disp_final.shape[1:3] == (1, 1):
            disp_final = piv.smooth(time_s, disp_final, lam=lam, type=int)

    return disp_glo, disp_nbs, disp_final


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

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

    run_id = timestamp_str()
    run_dir = resume_run_dir or init_run_dir(Path(cfg.OUTPUT_DIR), run_id)

    # Pair mapping for CSV row -> image filenames (kept separate to avoid ballooning).
    pairs = [
        (
            i,
            _relpath_or_name(cfg.IMAGE_LIST[i], base_dir=cfg.IMAGE_DIR),
            _relpath_or_name(cfg.IMAGE_LIST[i + 1], base_dir=cfg.IMAGE_DIR),
        )
        for i in range(cfg.NR_IMAGES - 1)
    ]
    write_pairs_csv(run_dir / "pairs.csv", pairs)

    frames = _frames_for_timebase()
    time_s = piv.get_time(frames, float(cfg.TIMESTEP_S))

    imgs: np.ndarray | None = None
    prev_disp_final: np.ndarray | None = None

    for pass_i in range(cfg.NR_PASSES):
        pass_1b = pass_i + 1
        paths = pass_paths(run_dir, pass_1b)

        win_pos: np.ndarray | None = None

        n_wins = tuple(_per_pass(cfg.NR_WINDOWS, pass_i))
        n_wy, n_wx = int(n_wins[0]), int(n_wins[1])
        n_pairs = len(time_s)

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

        # Resume: if post exists, load and continue.
        if paths.post_csv.exists():
            _, _, _, disp_final = load_postprocessed_csv(
                paths.post_csv,
                n_pairs=n_pairs,
                n_wy=n_wy,
                n_wx=n_wx,
            )
            prev_disp_final = disp_final
            continue

        # Load peak-detection checkpoint if present.
        if paths.peaks_csv_gz.exists():
            disp_unf, int_unf = load_unfiltered_peaks_csv_gz(
                paths.peaks_csv_gz,
                n_pairs=n_pairs,
                n_wy=n_wy,
                n_wx=n_wx,
                n_peaks=int(_per_pass(cfg.NR_PEAKS, pass_i)),
            )
        else:
            # Correlation needs images; load once and reuse.
            if imgs is None:
                imgs = _load_and_preprocess_images()

            if pass_i == 0:
                shifts = None
            else:
                if prev_disp_final is None:
                    raise RuntimeError(f"Pass {pass_1b} needs previous displacement to compute shifts")
                shifts = piv.disp2shift((n_wy, n_wx), prev_disp_final)

            corrs = piv.calc_corrs(
                imgs,
                n_wins=(n_wy, n_wx),
                shifts=shifts,
                overlap=float(_per_pass(cfg.WINDOW_OVERLAP, pass_i)),
                ds_fac=int(_per_pass(cfg.DOWNSAMPLE_FACTOR, pass_i)),
            )
            corrs_sum = piv.sum_corrs(
                corrs,
                int(_per_pass(cfg.CORRS_TO_SUM, pass_i)),
                n_wins=(n_wy, n_wx),
                shifts=shifts,
            )
            disp_unf, int_unf = piv.find_disps(
                corrs_sum,
                n_wins=(n_wy, n_wx),
                shifts=shifts,
                n_peaks=int(_per_pass(cfg.NR_PEAKS, pass_i)),
                ds_fac=int(_per_pass(cfg.DOWNSAMPLE_FACTOR, pass_i)),
                min_dist=int(_per_pass(cfg.MIN_PEAK_DISTANCE, pass_i)),
            )

            # Window positions (for plotting) from first image.
            _, win_pos = piv.split_n_shift(
                imgs[0],
                (n_wy, n_wx),
                overlap=float(_per_pass(cfg.WINDOW_OVERLAP, pass_i)),
            )

            write_unfiltered_peaks_csv_gz(
                paths.peaks_csv_gz,
                disp_unf=disp_unf,
                int_unf=int_unf,
            )

        # Postprocessing stage and checkpoint.
        disp_glo, disp_nbs, disp_final = _postprocess_pass(
            pass_index_0b=pass_i,
            disp_unf=disp_unf,
            time_s=time_s,
        )

        write_postprocessed_csv(
            paths.post_csv,
            time_s=time_s,
            disp_glo=disp_glo,
            disp_nbs=disp_nbs,
            disp_final=disp_final,
        )

        if win_pos is not None:
            write_win_pos_csv(paths.win_pos_csv, win_pos=win_pos)

        write_meta_json(paths.meta_json, meta)
        prev_disp_final = disp_final

    print(f"Done. Run directory: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
