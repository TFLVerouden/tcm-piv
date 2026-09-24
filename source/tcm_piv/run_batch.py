"""Batch entrypoint for tcm-piv.

Run this file directly from VS Code to process every immediate subfolder in a
selected top-level directory. It prompts for:
- the top-level folder containing the video subfolders,
- the shared TOML config.

Subfolders that share the same first four characters are treated as one video
group. Each subfolder in a group is still processed separately so the machine
only handles a smaller batch at a time, and the resulting flow-rate series are
stitched together in alphabetical order afterwards.

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
from tcm_utils.io_utils import beep, prompt_yes_no


def run_batch(
    *,
    top_dir: str | Path | None = None,
    config_file: str | Path | None = None,
) -> list[Path]:
    """Run the pipeline for grouped subfolders in a top-level folder."""

    batch_run_id = timestamp_str()
    batch_start_s = perf_counter()
    top_dir_path = _resolve_batch_directory(top_dir)
    config_path = _resolve_batch_config_file(config_file)

    video_dirs = [p for p in natsorted(
        top_dir_path.iterdir()) if p.is_dir() and not p.name.startswith(".")]
    if not video_dirs:
        raise RuntimeError(f"No subfolders found in {top_dir_path}")

    run_dirs: list[Path] = []

    groups = _group_video_dirs(video_dirs)

    print(f"\nBatch input folder: {top_dir_path}")
    print(f"Batch run id: {batch_run_id}")
    print(f"Batch config file: {config_path}")

    print("\nProposed stitching plan:")
    for group_key, group_dirs in groups:
        print(f"\n-  Stitch group '{group_key}':")
        for video_dir in group_dirs:
            print(f"    - {video_dir.name}")

    print("\nEach group will be processed in the listed order and stitched into")
    print("one flow-rate result.")

    if not prompt_yes_no("Continue with this plan? (press ENTER to confirm)"):
        print("\nBatch cancelled; no folders were processed.")
        return []

    run_dirs: list[Path] = []
    stitched_flow_series: list[tuple[str, np.ndarray, np.ndarray]] = []
    manifest_rows: list[tuple[str, str, str, str, str, str]] = []

    try:
        for group_key, group_dirs in groups:

            print(
                f"================================================================================")
            print(f"\nBatch group: {group_key}")
            print(f"  folders: {[p.name for p in group_dirs]}")

            group_run_rows: list[tuple[str, str, str, str, str, str]] = []
            grouped_series: list[
                tuple[str, np.ndarray, np.ndarray, np.ndarray]
            ] = []

            for video_dir in group_dirs:
                camera_file = _find_camera_file(video_dir)
                if camera_file is None:
                    print(
                        f"Skipping {video_dir} because no camera file was found.")
                    continue
                output_dir = video_dir / f"piv_run_{batch_run_id}"
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
                pair_index, time_s, flow_m3_s, flow_ls = _read_flow_rate_csv(
                    flow_csv)
                grouped_series.append(
                    (video_dir.name, pair_index, time_s, flow_ls))
                group_run_rows.append(
                    (
                        video_dir.name,
                        str(run_dir),
                        str(flow_csv),
                        str(config_path),
                        batch_run_id,
                        str(top_dir_path),
                    )
                )

            if not grouped_series:
                print(
                    f"No completed runs in group {group_key}; skipping stitch.")
                continue

            stitched_time_s, stitched_flow_ls, stitched_rows = _stitch_flow_rate_series(
                grouped_series,
            )

            # One plotting series per stitched experiment, e.g. P-001.
            stitched_flow_series.append(
                (group_key.upper(), stitched_time_s, stitched_flow_ls)
            )
            stitched_flow_path = (
                top_dir_path /
                f"{batch_run_id}_{group_key}_stitched_flow_rate.csv"
            )
            _write_stitched_flow_rate_csv(stitched_flow_path, stitched_rows)
            print(f"Batch stitched flow data: {stitched_flow_path}")

            # Retain one batch-wide manifest, as in the original version.
            manifest_rows.extend(group_run_rows)

        manifest_path = top_dir_path / f"{batch_run_id}_piv_result.csv"
        _write_flow_rate_manifest(manifest_path, manifest_rows)
        print(f"\nBatch manifest: {manifest_path}")

        comparison_plot_path = top_dir_path / \
            f"{batch_run_id}_piv_comparison.pdf"
        viz.plot_flow_rate_series(
            stitched_flow_series,
            title=top_dir_path.name,
            output_path=comparison_plot_path,
        )
        print(f"Batch comparison plot: {comparison_plot_path}")

        return run_dirs
    finally:
        elapsed_s = perf_counter() - batch_start_s
        print(f"Batch elapsed time: {elapsed_s:.1f} s")
        beep()


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


def _find_camera_file(video_dir: Path) -> Path | None:
    candidates = [p for p in natsorted(
        video_dir.rglob("*.cihx")) if p.is_file()]
    if not candidates:
        print(f"No .cihx file found in {video_dir}")
        return None
    if len(candidates) > 1:
        print(
            f"Warning: found {len(candidates)} .cihx files in {video_dir}; using {candidates[0]}"
        )
    return candidates[0]


def _group_video_dirs(video_dirs: list[Path]) -> list[tuple[str, list[Path]]]:
    grouped: dict[str, list[Path]] = {}
    for video_dir in video_dirs:
        group_key = _video_group_key(video_dir)
        grouped.setdefault(group_key, []).append(video_dir)

    grouped_dirs: list[tuple[str, list[Path]]] = []
    for group_key in natsorted(grouped):
        grouped_dirs.append((group_key, natsorted(grouped[group_key])))
    return grouped_dirs


def _video_group_key(video_dir: Path) -> str:
    name = video_dir.name.strip()
    return name[:5].lower() if len(name) >= 5 else name.lower()


def _read_flow_rate_csv(
    flow_csv: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.genfromtxt(flow_csv, delimiter=",", names=True)
    if data.size == 0:
        raise RuntimeError(f"No data found in {flow_csv}")
    if getattr(data, "ndim", 0) == 0:
        data = np.array([data], dtype=data.dtype)
    return (
        np.asarray(data["pair_index"]),
        np.asarray(data["time_s"]),
        np.asarray(data["flow_rate_m3_s"]),
        np.asarray(data["flow_rate_L_s"]),
    )


def _stitch_flow_rate_series(
    series: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, list[tuple[str, int, int, float, float, float]]]:
    stitched_time_parts: list[np.ndarray] = []
    stitched_flow_parts: list[np.ndarray] = []
    stitched_rows: list[tuple[str, int, int, float, float, float]] = []

    time_offset = 0.0
    last_step: float | None = None
    stitched_pair_index = 0

    for label, pair_index, time_s, flow_ls in series:
        t = np.asarray(time_s).reshape(-1)
        q = np.asarray(flow_ls).reshape(-1)
        pair = np.asarray(pair_index).reshape(-1)
        if t.shape != q.shape or t.shape != pair.shape:
            raise ValueError(
                f"time_s, pair_index, and flow_ls must match for {label!r}, got {pair.shape}, {t.shape}, {q.shape}"
            )
        if t.size == 0:
            continue

        local_t = t - t[0]
        if stitched_time_parts:
            local_t = local_t + time_offset

        stitched_time_parts.append(local_t)
        stitched_flow_parts.append(q)

        for source_pair_index, source_time_s, flow_value in zip(pair, local_t, q):
            stitched_rows.append(
                (
                    label,
                    int(source_pair_index),
                    stitched_pair_index,
                    float(source_time_s),
                    float(flow_value) / 1000.0,
                    float(flow_value),
                )
            )
            stitched_pair_index += 1

        step = _estimate_time_step(local_t, fallback=last_step)
        if step is not None:
            last_step = step
            time_offset = float(local_t[-1] + step)
        else:
            time_offset = float(local_t[-1])

    if not stitched_time_parts:
        raise RuntimeError("No flow-rate data was available to stitch")

    return (
        np.concatenate(stitched_time_parts),
        np.concatenate(stitched_flow_parts),
        stitched_rows,
    )


def _estimate_time_step(time_s: np.ndarray, *, fallback: float | None = None) -> float | None:
    t = np.asarray(time_s).reshape(-1)
    if t.size < 2:
        return fallback
    diffs = np.diff(t)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return fallback
    step = float(np.median(diffs))
    if step <= 0:
        return fallback
    return step


def _write_stitched_flow_rate_csv(
    path: Path,
    rows: list[tuple[str, int, int, float, float, float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow([
            "source_subfolder",
            "source_pair_index",
            "stitched_pair_index",
            "time_s",
            "flow_rate_m3_s",
            "flow_rate_L_s",
        ])
        writer.writerows(rows)


def stitch_existing_batch(
    *,
    batch_run_id: str,
    top_dir: str | Path | None = None,
) -> list[Path]:
    """Stitch and plot flow data from an already completed batch run."""

    top_dir_path = _resolve_batch_directory(top_dir)

    video_dirs = [
        path for path in natsorted(top_dir_path.iterdir())
        if path.is_dir() and not path.name.startswith(".")
    ]
    groups = _group_video_dirs(video_dirs)

    print(f"\nRestitching existing batch: {batch_run_id}")
    print(f"Top-level folder: {top_dir_path}")
    print("\nProposed stitching plan:")

    for group_key, group_dirs in groups:
        print(f"\n- Stitch group '{group_key}':")
        for video_dir in group_dirs:
            flow_csv = video_dir / f"piv_run_{batch_run_id}" / "flow_rate.csv"
            status = "found" if flow_csv.is_file() else "MISSING"
            print(f"    - {video_dir.name}: {status}")

    if not prompt_yes_no("Create restitched CSVs and comparison plot?"):
        print("Restitching cancelled.")
        return []

    stitched_flow_series: list[tuple[str, np.ndarray, np.ndarray]] = []
    output_paths: list[Path] = []

    for group_key, group_dirs in groups:
        grouped_series: list[
            tuple[str, np.ndarray, np.ndarray, np.ndarray]
        ] = []

        for video_dir in group_dirs:
            flow_csv = video_dir / f"piv_run_{batch_run_id}" / "flow_rate.csv"

            if not flow_csv.is_file():
                print(f"Skipping missing flow file: {flow_csv}")
                continue

            pair_index, time_s, _, flow_ls = _read_flow_rate_csv(flow_csv)
            grouped_series.append(
                (video_dir.name, pair_index, time_s, flow_ls)
            )

        if not grouped_series:
            print(f"No existing flow data for {group_key}; skipping.")
            continue

        stitched_time_s, stitched_flow_ls, stitched_rows = (
            _stitch_flow_rate_series(grouped_series)
        )

        stitched_path = (
            top_dir_path
            / f"{batch_run_id}_{group_key}_restitched_flow_rate.csv"
        )
        _write_stitched_flow_rate_csv(stitched_path, stitched_rows)
        output_paths.append(stitched_path)

        stitched_flow_series.append(
            (group_key.upper(), stitched_time_s, stitched_flow_ls)
        )
        print(f"Restitched flow data: {stitched_path}")

    if not stitched_flow_series:
        raise RuntimeError("No existing flow-rate CSV files were found.")

    comparison_plot_path = (
        top_dir_path / f"{batch_run_id}_restitched_comparison.pdf"
    )
    viz.plot_flow_rate_series(
        stitched_flow_series,
        title=top_dir_path.name,
        output_path=comparison_plot_path,
    )
    output_paths.append(comparison_plot_path)
    print(f"Restitched comparison plot: {comparison_plot_path}")

    return output_paths


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
            "top_dir_path",
        ])
        writer.writerows(rows)


def main() -> None:
    stitch_existing_batch(batch_run_id="260914_132007")


if __name__ == "__main__":
    main()
