"""Checkpoint read/write utilities for tcm-piv.

Design goals:
- Two checkpoints per pass:
  1) After peak detection (unfiltered, all peaks) -> large -> CSV.gz
  2) After postprocessing (final + intermediates) -> human-readable -> CSV
- Keep CSVs self-describing via small JSON metadata files.
- Allow resuming without re-loading images when only postprocessing.

CSV conventions:
- Displacements are stored as (dy_px, dx_px) to match internal coordinate order.
- Rows are keyed by (pair_index, win_y, win_x) and optionally peak_k.
- Mapping from pair_index -> image filenames lives in a separate pairs.csv.
"""

from __future__ import annotations

import csv
import gzip
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PassCheckpointPaths:
    pass_dir: Path
    peaks_csv_gz: Path
    post_csv: Path
    win_pos_csv: Path
    meta_json: Path


class PassStage(Enum):
    """Checkpoint stage for a pass.

    - POST: postprocessed (final) checkpoint exists -> pass considered done.
    - UNFILTERED: unfiltered peaks checkpoint exists -> can run postprocessing only.
    - NONE: nothing exists -> full computation needed.
    """

    POST = "post"
    UNFILTERED = "unfiltered"
    NONE = "none"


def init_run_dir(output_dir: Path, run_id: str) -> Path:
    run_dir = output_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_pairs_csv(pairs_csv: Path, pairs: list[tuple[int, str, str]]) -> None:
    pairs_csv.parent.mkdir(parents=True, exist_ok=True)
    with pairs_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["pair_index", "image0", "image1"])
        writer.writerows(pairs)


def pass_paths(run_dir: Path, pass_index_1b: int) -> PassCheckpointPaths:
    pass_dir = run_dir / f"pass_{pass_index_1b:02d}"
    pass_dir.mkdir(parents=True, exist_ok=True)

    return PassCheckpointPaths(
        pass_dir=pass_dir,
        peaks_csv_gz=pass_dir / f"pass_{pass_index_1b:02d}_unfiltered.csv.gz",
        post_csv=pass_dir / f"pass_{pass_index_1b:02d}_post.csv",
        win_pos_csv=pass_dir / f"pass_{pass_index_1b:02d}_win_pos.csv",
        meta_json=pass_dir / f"pass_{pass_index_1b:02d}_meta.json",
    )


def pass_stage(paths: PassCheckpointPaths) -> PassStage:
    if paths.post_csv.exists():
        return PassStage.POST
    if paths.peaks_csv_gz.exists():
        return PassStage.UNFILTERED
    return PassStage.NONE


def load_interpolated_mask_csv(
    path: Path,
    *,
    n_pairs: int,
    n_wy: int,
    n_wx: int,
) -> np.ndarray:
    """Load `interpolated_mask.csv` as a boolean array of shape (n_pairs,n_wy,n_wx)."""

    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim != 2 or data.shape[1] != 4:
        raise ValueError(
            f"Unexpected interpolated mask CSV shape {data.shape} in {path}")

    mask_flat = data[:, 3].astype(bool)
    if mask_flat.size != n_pairs * n_wy * n_wx:
        raise ValueError(
            f"Unexpected interpolated mask size {mask_flat.size}, expected {n_pairs*n_wy*n_wx}"
        )
    return mask_flat.reshape(n_pairs, n_wy, n_wx)


def write_meta_json(path: Path, meta: dict[str, Any]) -> None:
    path.write_text(json.dumps(meta, indent=2, sort_keys=True),
                    encoding="utf-8")


def write_unfiltered_peaks_csv_gz(
    path: Path,
    *,
    disp_unf: np.ndarray,
    int_unf: np.ndarray,
) -> None:
    """Write unfiltered multi-peak displacement results.

    Expected shapes:
    - disp_unf: (n_pairs, n_wy, n_wx, n_peaks, 2)
    - int_unf:  (n_pairs, n_wy, n_wx, n_peaks)
    """

    if disp_unf.ndim != 5 or disp_unf.shape[-1] != 2:
        raise ValueError(
            f"disp_unf must be 5D (..., n_peaks, 2), got {disp_unf.shape}")
    if int_unf.shape != disp_unf.shape[:-1]:
        raise ValueError(
            f"int_unf must match disp_unf without last dim. Got int_unf={int_unf.shape}, disp_unf={disp_unf.shape}"
        )

    n_pairs, n_wy, n_wx, n_peaks, _ = disp_unf.shape

    pair_index = np.repeat(
        np.arange(n_pairs, dtype=np.int64), n_wy * n_wx * n_peaks)
    win_y = np.tile(
        np.repeat(np.arange(n_wy, dtype=np.int64), n_wx * n_peaks), n_pairs)
    win_x = np.tile(
        np.repeat(np.arange(n_wx, dtype=np.int64), n_peaks), n_pairs * n_wy)
    peak_k = np.tile(np.arange(n_peaks, dtype=np.int64), n_pairs * n_wy * n_wx)

    dy = disp_unf[..., 0].reshape(-1)
    dx = disp_unf[..., 1].reshape(-1)
    intensity = int_unf.reshape(-1)

    data = np.column_stack(
        [pair_index, win_y, win_x, peak_k, dy, dx, intensity])

    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="", encoding="utf-8") as fp:
        np.savetxt(
            fp,
            data,
            delimiter=",",
            header="pair_index,win_y,win_x,peak_k,dy_px,dx_px,intensity",
            comments="",
        )


def write_postprocessed_csv(
    path: Path,
    *,
    time_s: np.ndarray,
    disp_glo: np.ndarray | None,
    disp_nbs: np.ndarray | None,
    disp_final: np.ndarray,
) -> None:
    """Write postprocessed (single-peak) displacements.

    All displacement arrays are expected as:
    - (n_pairs, n_wy, n_wx, 2)
    time_s:
    - (n_pairs,)

    The output CSV is long-form with one row per (pair_index, win_y, win_x).
    """

    if disp_final.ndim != 4 or disp_final.shape[-1] != 2:
        raise ValueError(
            f"disp_final must be 4D (..., 2), got {disp_final.shape}")

    n_pairs, n_wy, n_wx, _ = disp_final.shape
    if time_s.shape != (n_pairs,):
        raise ValueError(
            f"time_s must have shape ({n_pairs},), got {time_s.shape}")

    def _validate_optional(name: str, arr: np.ndarray | None) -> np.ndarray:
        if arr is None:
            return np.full((n_pairs, n_wy, n_wx, 2), np.nan)
        if arr.shape != (n_pairs, n_wy, n_wx, 2):
            raise ValueError(
                f"{name} must have shape {(n_pairs, n_wy, n_wx, 2)}, got {arr.shape}")
        return arr

    disp_glo_v = _validate_optional("disp_glo", disp_glo)
    disp_nbs_v = _validate_optional("disp_nbs", disp_nbs)

    pair_index = np.repeat(np.arange(n_pairs, dtype=np.int64), n_wy * n_wx)
    win_y = np.tile(np.repeat(np.arange(n_wy, dtype=np.int64), n_wx), n_pairs)
    win_x = np.tile(np.arange(n_wx, dtype=np.int64), n_pairs * n_wy)
    time_rep = np.repeat(time_s, n_wy * n_wx)

    data = np.column_stack(
        [
            pair_index,
            win_y,
            win_x,
            time_rep,
            disp_glo_v[..., 0].reshape(-1),
            disp_glo_v[..., 1].reshape(-1),
            disp_nbs_v[..., 0].reshape(-1),
            disp_nbs_v[..., 1].reshape(-1),
            disp_final[..., 0].reshape(-1),
            disp_final[..., 1].reshape(-1),
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        np.savetxt(
            fp,
            data,
            delimiter=",",
            header=(
                "pair_index,win_y,win_x,time_s,"
                "glo_dy_px,glo_dx_px,"
                "nbs_dy_px,nbs_dx_px,"
                "final_dy_px,final_dx_px"
            ),
            comments="",
        )


def write_win_pos_csv(path: Path, *, win_pos: np.ndarray) -> None:
    """Write window center positions.

    win_pos shape:
    - (n_wy, n_wx, 2) where last dim is (y_px, x_px)
    """

    if win_pos.ndim != 3 or win_pos.shape[-1] != 2:
        raise ValueError(
            f"win_pos must be 3D (n_wy, n_wx, 2), got {win_pos.shape}")

    n_wy, n_wx, _ = win_pos.shape
    win_y = np.repeat(np.arange(n_wy, dtype=np.int64), n_wx)
    win_x = np.tile(np.arange(n_wx, dtype=np.int64), n_wy)

    data = np.column_stack(
        [win_y, win_x, win_pos[..., 0].reshape(-1), win_pos[..., 1].reshape(-1)])

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        np.savetxt(
            fp,
            data,
            delimiter=",",
            header="win_y,win_x,center_y_px,center_x_px",
            comments="",
        )


def load_unfiltered_peaks_csv_gz(
    path: Path,
    *,
    n_pairs: int,
    n_wy: int,
    n_wx: int,
    n_peaks: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Load unfiltered checkpoint from CSV.gz back into arrays."""

    with gzip.open(path, "rt", encoding="utf-8") as fp:
        data = np.loadtxt(fp, delimiter=",", skiprows=1)

    if data.ndim != 2 or data.shape[1] != 7:
        raise ValueError(
            f"Unexpected unfiltered CSV shape {data.shape} in {path}")

    dy = data[:, 4].reshape(n_pairs, n_wy, n_wx, n_peaks)
    dx = data[:, 5].reshape(n_pairs, n_wy, n_wx, n_peaks)
    intensity = data[:, 6].reshape(n_pairs, n_wy, n_wx, n_peaks)

    disp_unf = np.stack([dy, dx], axis=-1)
    return disp_unf, intensity


def load_postprocessed_csv(
    path: Path,
    *,
    n_pairs: int,
    n_wy: int,
    n_wx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load postprocessed checkpoint.

    Returns: (time_s, disp_glo, disp_nbs, disp_final)
    """

    data = np.loadtxt(path, delimiter=",", skiprows=1)
    if data.ndim != 2 or data.shape[1] != 10:
        raise ValueError(f"Unexpected post CSV shape {data.shape} in {path}")

    time_rep = data[:, 3]
    time_s = time_rep.reshape(n_pairs, n_wy, n_wx)[:, 0, 0]

    def _reshape_pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.stack([a.reshape(n_pairs, n_wy, n_wx), b.reshape(n_pairs, n_wy, n_wx)], axis=-1)

    disp_glo = _reshape_pair(data[:, 4], data[:, 5])
    disp_nbs = _reshape_pair(data[:, 6], data[:, 7])
    disp_final = _reshape_pair(data[:, 8], data[:, 9])

    return time_s, disp_glo, disp_nbs, disp_final
