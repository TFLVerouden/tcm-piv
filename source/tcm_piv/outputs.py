"""Output file writers for tcm-piv.

Each pass writes its final (postprocessed) results to disk:
- `pass_XX_post.csv`: single-peak displacement field (+ intermediates).
- `pass_XX_meta.json`: small metadata describing the pass's config/shapes.

The mapping from `pair_index -> image0,image1` is written once as `pairs.csv`
in the run root.

CSV conventions:
- Displacements are stored as (dy_px, dx_px) to match internal coordinate order.
- Rows are keyed by (pair_index, win_y, win_x).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PassPaths:
    pass_dir: Path
    post_csv: Path
    meta_json: Path


def init_run_dir(output_dir: Path, run_id: str, *, overwrite_runs: bool = False) -> Path:
    run_dir = output_dir if overwrite_runs else output_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_pairs_csv(pairs_csv: Path, pairs: list[tuple[int, str, str]]) -> None:
    pairs_csv.parent.mkdir(parents=True, exist_ok=True)
    with pairs_csv.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["pair_index", "image0", "image1"])
        writer.writerows(pairs)


def pass_paths(run_dir: Path, pass_index_1b: int) -> PassPaths:
    pass_dir = run_dir / f"pass_{pass_index_1b:02d}"
    pass_dir.mkdir(parents=True, exist_ok=True)

    return PassPaths(
        pass_dir=pass_dir,
        post_csv=pass_dir / f"pass_{pass_index_1b:02d}_post.csv",
        meta_json=pass_dir / f"pass_{pass_index_1b:02d}_meta.json",
    )


def write_meta_json(path: Path, meta: dict[str, Any]) -> None:
    path.write_text(json.dumps(meta, indent=2, sort_keys=True),
                    encoding="utf-8")


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
