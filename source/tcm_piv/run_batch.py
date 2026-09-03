"""Batch entrypoint for tcm-piv.

Run this file directly from VS Code to process every immediate subfolder in a
selected top-level directory. It prompts for:
- the top-level folder containing the video subfolders,
- the shared TOML config.

For each subfolder it calls :func:`tcm_piv.run.run` with overrides so the
existing single-run pipeline remains the core implementation.
"""

from __future__ import annotations

import csv
from pathlib import Path
from time import perf_counter

import numpy as np
from natsort import natsorted
from tcm_utils.time_utils import timestamp_str

from tcm_piv.run import run
import tcm_piv.visualisation as viz
from tcm_utils.file_dialogs import ask_directory, ask_open_file


def run_batch(
    *,
    top_dir: str | Path | None = None,
    config_file: str | Path | None = None,
) -> list[Path]:
    """Run the pipeline for every immediate subfolder in a top-level folder."""

    batch_run_id = timestamp_str()
    batch_start_s = perf_counter()
    top_dir_path = _resolve_batch_directory(top_dir)
    config_path = _resolve_batch_config_file(config_file)

    video_dirs = [p for p in natsorted(
        top_dir_path.iterdir()) if p.is_dir() and not p.name.startswith(".")]
    if not video_dirs:
        raise RuntimeError(f"No subfolders found in {top_dir_path}")

    run_dirs: list[Path] = []
    flow_series: list[tuple[str, np.ndarray, np.ndarray]] = []
    manifest_rows: list[tuple[str, str, str, str, str, str]] = []

    print(f"\nBatch input folder: {top_dir_path}")
    print(f"Batch run id: {batch_run_id}")
    print(f"Batch config file: {config_path}")

    try:
        for video_dir in video_dirs:
            camera_file = _find_camera_file(video_dir)
            output_dir = video_dir / f"piv_run_{batch_run_id}"
            print(
                f"================================================================================")
            print(f"\nBatch folder: {video_dir.name}")
            print(f"  image_dir: {video_dir}")
            print(f"  camera_file: {camera_file}")
            print(f"  output_dir: {output_dir}")

            run_dir = run(
                config_file=config_path,
                config_overrides={
                    "source": {
                        "image_dir": video_dir,
                        "output_dir": output_dir,
                        "overwrite_runs": True,
                    },
                    "camera": {
                        "camera_dir": camera_file,
                    },
                },
            )
            run_dirs.append(run_dir)

            flow_csv = run_dir / "flow_rate.csv"
            time_s, flow_ls = _read_flow_rate_csv(flow_csv)
            flow_series.append((video_dir.name, time_s, flow_ls))
            manifest_rows.append(
                (
                    video_dir.name,
                    str(run_dir),
                    str(flow_csv),
                    str(config_path),
                    batch_run_id,
                )
            )

        manifest_path = top_dir_path / f"{batch_run_id}_flow_rate_manifest.csv"
        _write_flow_rate_manifest(manifest_path, manifest_rows)
        print(f"\nBatch manifest: {manifest_path}")

        plot_path = top_dir_path / f"{batch_run_id}_flow_rate_comparison.png"
        viz.plot_flow_rate_series(
            flow_series,
            title=top_dir_path.name,
            output_path=plot_path,
        )
        print(f"Batch plot: {plot_path}")

        return run_dirs
    finally:
        elapsed_s = perf_counter() - batch_start_s
        print(f"Batch elapsed time: {elapsed_s:.1f} s")


def _resolve_batch_directory(top_dir: str | Path | None) -> Path:
    if top_dir is not None:
        top_dir_path = Path(top_dir)
    else:
        selected = ask_directory(
            key="batch_top_dir",
            title="Select the top-level folder containing the video subfolders",
            default_dir=Path.cwd(),
        )
        if selected is None:
            raise RuntimeError("No top-level folder selected; aborting.")
        top_dir_path = Path(selected)

    if not top_dir_path.is_dir():
        raise NotADirectoryError(
            f"Top-level folder does not exist: {top_dir_path}")
    return top_dir_path


def _resolve_batch_config_file(config_file: str | Path | None) -> Path:
    if config_file is not None:
        config_path = Path(config_file)
        if not config_path.is_file():
            raise FileNotFoundError(
                f"Config file does not exist: {config_path}")
        return config_path

    selected_path = ask_open_file(
        key="batch_config_file",
        title="Select the base configuration TOML file",
        filetypes=(("TOML files", "*.toml"), ("All files", "*.*")),
    )
    if selected_path is None:
        raise RuntimeError("No configuration file selected; aborting.")
    return Path(selected_path)


def _find_camera_file(video_dir: Path) -> Path:
    candidates = [p for p in natsorted(
        video_dir.rglob("*.cihx")) if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No .cihx file found in {video_dir}")
    if len(candidates) > 1:
        print(
            f"Warning: found {len(candidates)} .cihx files in {video_dir}; using {candidates[0]}"
        )
    return candidates[0]


def _read_flow_rate_csv(flow_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(flow_csv, delimiter=",", names=True)
    if data.size == 0:
        raise RuntimeError(f"No data found in {flow_csv}")
    if getattr(data, "ndim", 0) == 0:
        data = np.array([data], dtype=data.dtype)
    return np.asarray(data["time_s"]), np.asarray(data["flow_rate_L_s"])


def _write_flow_rate_manifest(path: Path, rows: list[tuple[str, str, str, str, str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow([
            "subfolder",
            "run_dir",
            "flow_rate_csv",
            "config_file",
            "batch_run_id",
        ])
        writer.writerows(rows)


def main() -> None:
    run_batch()


if __name__ == "__main__":
    main()
