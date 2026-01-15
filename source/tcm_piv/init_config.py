"""Configuration loading for tcm-piv.

This module provides :func:`load_config`, which:
- loads a user TOML config,
- deep-merges it over packaged defaults,
- normalizes types and broadcasts per-pass parameters,
- resolves/ensures required input files (camera + calibration),
- and returns a single :class:`Config` object.

The end-to-end pipeline in :mod:`tcm_piv.run` is the primary consumer.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from copy import deepcopy
from pathlib import Path
import shutil
import tomllib
from typing import Any, Callable

import tifffile
from natsort import natsorted

from tcm_piv.preprocessing import crop, generate_background
from tcm_utils.camera_calibration import ensure_calibration
from tcm_utils.file_dialogs import ask_open_file
from tcm_utils.io_utils import ensure_path, load_images, load_metadata, prompt_yes_no
from tcm_utils.read_cihx import ensure_cihx_processed
from tcm_utils.time_utils import timestamp_str


@dataclass(frozen=True)
class Config:
    # Core
    config_path: Path
    nr_passes: int

    # [source]
    image_dir: Path
    output_dir: Path
    frames_to_use: str | list[int]
    image_list: list[str]
    nr_images: int

    # [camera]
    camera_dir: Path
    calib_dir: Path
    calib_spacing_mm: float
    timestamp: str
    framerate_hz: int
    timestep_s: float
    shutterspeed_ns: int
    image_width_px: int
    image_height_px: int
    image_width_m: float
    image_height_m: float
    scale_m_per_px: float

    # [preprocessing]
    ds_factor: list[int]
    background_dir: str
    crop_roi: tuple[int, int, int, int]

    # [correlation]
    n_corrs_to_sum: list[int]
    n_windows: list[tuple[int, int]]
    window_overlap: list[float]

    # [displacement]
    n_peaks: list[int]
    min_peak_dist_px: list[int]

    # [postprocessing]
    outlier_filter_mode: list[str]
    max_velocity_vy_vx_m_s: list[tuple[float, float]]
    flow_direction: str
    extra_vel_dim_m: float
    nb_size_tyx: list[tuple[int, int, int]]
    interp_nb_size_tyx: list[tuple[int, int, int] | None]
    nb_threshold: list[int | tuple[int, int]]
    time_smooth_lam: list[float]

    # [visualisation]
    plot_model: bool
    model_gender: str
    model_mass: float
    model_height: float
    plot_global_filters: bool
    plot_correlations: bool
    plot_window_layout: list[bool]
    plot_flow_rate: bool
    export_velocity_profiles_pdf: bool


def _optional_path(value: Any) -> Path | None:
    s = "" if value is None else str(value).strip()
    return Path(s) if s else None


def _parse_source(source: dict[str, Any]) -> tuple[Path, Path, str | list[int], list[str]]:
    frames_to_use = source["frames_to_use"]
    if not (frames_to_use == "all" or isinstance(frames_to_use, list)):
        raise ValueError(
            "source.frames_to_use must be 'all' or an array of integers")
    if isinstance(frames_to_use, list):
        frames_to_use = [int(v) for v in frames_to_use]

    image_dir = Path(ensure_path(
        str(source["image_dir"]), "image_dir", title="Select image directory"))
    output_dir = Path(
        ensure_path(
            str(source["output_dir"]),
            "output_dir",
            title="Select output directory",
            default_dir=image_dir,
        )
    )

    image_list = _get_image_list(image_dir, frames_to_use)
    if len(image_list) < 2:
        raise RuntimeError(
            f"Found only {len(image_list)} images in {image_dir}. At least 2 images are required."
        )

    return image_dir, output_dir, frames_to_use, image_list


def _parse_camera(camera: dict[str, Any], *, output_dir: Path) -> tuple[
    float,
    Path,
    Path,
    str,
    int,
    float,
    int,
    int,
    int,
    float,
    float,
    float,
]:
    calib_spacing_mm = float(camera["calib_spacing_mm"])

    camera_dir = Path(
        ensure_cihx_processed(
            _optional_path(camera["camera_dir"]),
            output_dir=output_dir / "camera_metadata",
        )
    )
    calib_dir = Path(
        ensure_calibration(
            _optional_path(camera["calib_dir"]),
            distance_mm=calib_spacing_mm,
            output_dir=output_dir / "calibration",
        )
    )

    camera_metadata = load_metadata(camera_dir)
    timestamp = str(camera_metadata.get("timestamp"))
    camera_meta = camera_metadata.get("camera_metadata", {})
    framerate_hz = int(camera_meta.get("recordRate"))
    timestep_s = 1.0 / framerate_hz
    shutterspeed_ns = int(camera_meta.get("shutterSpeedNsec"))
    image_width_px = int(camera_meta.get("resolution", {}).get("width"))
    image_height_px = int(camera_meta.get("resolution", {}).get("height"))

    calib_metadata = load_metadata(calib_dir)
    image_width_m = float(
        calib_metadata.get("calibration", {}).get(
            "image_size_m", {}).get("width")
    )
    image_height_m = float(
        calib_metadata.get("calibration", {}).get(
            "image_size_m", {}).get("height")
    )
    scale_m_per_px = float(calib_metadata.get(
        "calibration", {}).get("scale_m_per_px"))

    return (
        calib_spacing_mm,
        camera_dir,
        calib_dir,
        timestamp,
        framerate_hz,
        timestep_s,
        shutterspeed_ns,
        image_width_px,
        image_height_px,
        image_width_m,
        image_height_m,
        scale_m_per_px,
    )


def _parse_preprocessing(
    preprocessing: dict[str, Any],
    *,
    nr_passes: int,
    image_list: list[str],
    output_dir: Path,
) -> tuple[list[int], tuple[int, int, int, int], str, str]:
    ds_factor = _per_pass(
        preprocessing["ds_factor"],
        nr_passes=nr_passes,
        element_parser=lambda v: int(v),
    )
    # type: ignore[assignment]
    crop_roi = _as_int_tuple(preprocessing["crop_roi"], length=4)

    bg_cfg_raw = preprocessing["background_dir"]
    bg_cfg_str = "" if bg_cfg_raw is None else str(bg_cfg_raw)
    bg_cfg_norm = bg_cfg_str.strip().lower()
    skip_background_generation = bg_cfg_norm in {"none", "null"}
    background_dir_for_config = bg_cfg_str if skip_background_generation else ""

    background_dir = "" if skip_background_generation else bg_cfg_str
    if not background_dir and not skip_background_generation:
        background_dir = _maybe_generate_background(
            image_paths=image_list,
            output_dir=output_dir,
            crop_roi=crop_roi,
            image_count=len(image_list),
        )

    return ds_factor, crop_roi, background_dir_for_config, background_dir


def _parse_correlation(
    correlation: dict[str, Any], *, nr_passes: int
) -> tuple[list[int], list[tuple[int, int]], list[float]]:
    n_corrs_to_sum = _per_pass(
        correlation["n_corrs_to_sum"],
        nr_passes=nr_passes,
        element_parser=lambda v: int(v),
    )
    n_windows = _per_pass(
        correlation["n_windows"],
        nr_passes=nr_passes,
        element_parser=lambda v: _as_int_tuple(v, length=2),
        tuple_len=2,
    )
    window_overlap = _per_pass(
        correlation["window_overlap"],
        nr_passes=nr_passes,
        element_parser=lambda v: float(v),
    )
    return (
        [int(v) for v in n_corrs_to_sum],
        [(int(v[0]), int(v[1])) for v in n_windows],
        [float(v) for v in window_overlap],
    )


def _parse_displacement(
    displacement: dict[str, Any], *, nr_passes: int
) -> tuple[list[int], list[int]]:
    n_peaks = _per_pass(
        displacement["n_peaks"],
        nr_passes=nr_passes,
        element_parser=lambda v: int(v),
    )
    min_peak_dist_px = _per_pass(
        displacement["min_peak_dist_px"],
        nr_passes=nr_passes,
        element_parser=lambda v: int(v),
    )
    return [int(v) for v in n_peaks], [int(v) for v in min_peak_dist_px]


def _parse_postprocessing(
    postprocessing: dict[str, Any], *, nr_passes: int
) -> tuple[
    list[str],
    list[tuple[float, float]],
    str,
    float,
    list[tuple[int, int, int]],
    list[tuple[int, int, int] | None],
    list[int | tuple[int, int]],
    list[float],
]:
    outlier_filter_mode = _per_pass(
        postprocessing["outlier_filter_mode"],
        nr_passes=nr_passes,
        element_parser=lambda v: str(v).strip().lower(),
    )
    allowed_outlier_modes = {"semicircle_rect", "circle"}
    for mode in outlier_filter_mode:
        if mode not in allowed_outlier_modes:
            raise ValueError(
                "postprocessing.outlier_filter_mode must be 'semicircle_rect' or 'circle'")

    max_velocity_vy_vx_m_s = _per_pass(
        postprocessing["max_velocity_vy_vx"],
        nr_passes=nr_passes,
        element_parser=lambda v: _as_float_tuple(v, length=2),
        tuple_len=2,
    )

    flow_direction = str(postprocessing["flow_direction"]).strip().lower()
    if flow_direction not in {"x", "y"}:
        raise ValueError("postprocessing.flow_direction must be 'x' or 'y'")
    extra_vel_dim_m = float(postprocessing["extra_vel_dim_m"])

    nb_size_tyx = _per_pass(
        postprocessing["nb_size_tyx"],
        nr_passes=nr_passes,
        element_parser=lambda v: _as_int_tuple(v, length=3),
        tuple_len=3,
    )

    def _parse_interp_nb(v: Any) -> tuple[int, int, int] | None:
        if isinstance(v, str) and v.strip().lower() in {"none", "null"}:
            return None
        return _as_int_tuple(v, length=3)  # type: ignore[return-value]

    interp_nb_size_tyx = _per_pass(
        postprocessing["interp_nb_size_tyx"],
        nr_passes=nr_passes,
        element_parser=_parse_interp_nb,
        tuple_len=3,
    )

    def _parse_threshold(v: Any) -> int | tuple[int, int]:
        if isinstance(v, list):
            if len(v) != 2:
                raise ValueError("nb_threshold tuple must have length 2")
            return (int(v[0]), int(v[1]))
        return int(v)

    thr_value = postprocessing["nb_threshold"]
    if (
        isinstance(thr_value, list)
        and _is_scalar_sequence(thr_value)
        and len(thr_value) == 2
        and all(isinstance(x, (int, float)) for x in thr_value)
    ):
        nb_threshold = [_parse_threshold(thr_value) for _ in range(nr_passes)]
    else:
        nb_threshold = _per_pass(
            thr_value,
            nr_passes=nr_passes,
            element_parser=_parse_threshold,
        )

    time_smooth_lam = _per_pass(
        postprocessing["time_smooth_lam"],
        nr_passes=nr_passes,
        element_parser=lambda v: float(v),
    )

    return (
        [str(m) for m in outlier_filter_mode],
        [(float(v[0]), float(v[1])) for v in max_velocity_vy_vx_m_s],
        flow_direction,
        extra_vel_dim_m,
        [(int(v[0]), int(v[1]), int(v[2])) for v in nb_size_tyx],
        interp_nb_size_tyx,
        nb_threshold,
        [float(v) for v in time_smooth_lam],
    )


def _parse_visualisation(
    visualisation: dict[str, Any], *, nr_passes: int
) -> tuple[bool, str, float, float, bool, bool, list[bool], bool, bool]:
    plot_model = bool(visualisation["plot_model"])
    model_gender = str(visualisation["model_gender"])
    model_mass = float(visualisation["model_mass"])
    model_height = float(visualisation["model_height"])
    plot_global_filters = bool(visualisation["plot_global_filters"])
    plot_correlations = bool(visualisation["plot_correlations"])
    plot_window_layout = _per_pass(
        visualisation["plot_window_layout"],
        nr_passes=nr_passes,
        element_parser=lambda v: bool(v),
    )
    plot_flow_rate = bool(visualisation["plot_flow_rate"])
    export_velocity_profiles_pdf = bool(
        visualisation["export_velocity_profiles_pdf"])
    return (
        plot_model,
        model_gender,
        model_mass,
        model_height,
        plot_global_filters,
        plot_correlations,
        [bool(v) for v in plot_window_layout],
        plot_flow_rate,
        export_velocity_profiles_pdf,
    )


def load_config(config_file: Path | str | None) -> Config:
    """Load TOML config, apply defaults, normalize settings, and return a Config."""

    config_path = _resolve_config_path(config_file)
    print(f"Reading configuration from: {config_path}.")

    with config_path.open("rb") as fp:
        user_cfg = tomllib.load(fp)

    defaults = _read_packaged_default_config()
    merged = _deep_merge(defaults, user_cfg)
    original_snapshot = deepcopy(merged)

    nr_passes = int(merged["nr_passes"])
    if nr_passes < 1:
        raise ValueError("nr_passes must be >= 1")

    source = merged["source"]
    camera = merged["camera"]
    preprocessing = merged["preprocessing"]
    correlation = merged["correlation"]
    displacement = merged["displacement"]
    postprocessing = merged["postprocessing"]
    visualisation = merged["visualisation"]

    image_dir, output_dir, frames_to_use, image_list = _parse_source(source)
    print(f"Number of images to process: {len(image_list)}")
    nr_images = len(image_list)

    (
        calib_spacing_mm,
        camera_dir,
        calib_dir,
        timestamp,
        framerate_hz,
        timestep_s,
        shutterspeed_ns,
        image_width_px,
        image_height_px,
        image_width_m,
        image_height_m,
        scale_m_per_px,
    ) = _parse_camera(camera, output_dir=output_dir)

    ds_factor, crop_roi, background_dir_for_config, background_dir = _parse_preprocessing(
        preprocessing,
        nr_passes=nr_passes,
        image_list=image_list,
        output_dir=output_dir,
    )

    n_corrs_to_sum, n_windows, window_overlap = _parse_correlation(
        correlation, nr_passes=nr_passes
    )
    n_peaks, min_peak_dist_px = _parse_displacement(
        displacement, nr_passes=nr_passes)

    (
        outlier_filter_mode,
        max_velocity_vy_vx_m_s,
        flow_direction,
        extra_vel_dim_m,
        nb_size_tyx,
        interp_nb_size_tyx,
        nb_threshold,
        time_smooth_lam,
    ) = _parse_postprocessing(postprocessing, nr_passes=nr_passes)

    (
        plot_model,
        model_gender,
        model_mass,
        model_height,
        plot_global_filters,
        plot_correlations,
        plot_window_layout,
        plot_flow_rate,
        export_velocity_profiles_pdf,
    ) = _parse_visualisation(visualisation, nr_passes=nr_passes)

    _maybe_write_updated_config(
        config_path=config_path,
        original_snapshot=original_snapshot,
        image_dir=image_dir,
        output_dir=output_dir,
        frames_to_use=frames_to_use,
        camera_dir=camera_dir,
        calib_dir=calib_dir,
        calib_spacing_mm=calib_spacing_mm,
        ds_factor=ds_factor,
        background_dir_for_config=background_dir_for_config,
        background_dir=background_dir,
        crop_roi=crop_roi,
    )

    return Config(
        config_path=config_path,
        nr_passes=nr_passes,
        image_dir=image_dir,
        output_dir=output_dir,
        frames_to_use=frames_to_use,
        image_list=image_list,
        nr_images=nr_images,
        camera_dir=camera_dir,
        calib_dir=calib_dir,
        calib_spacing_mm=calib_spacing_mm,
        timestamp=timestamp,
        framerate_hz=framerate_hz,
        timestep_s=timestep_s,
        shutterspeed_ns=shutterspeed_ns,
        image_width_px=image_width_px,
        image_height_px=image_height_px,
        image_width_m=image_width_m,
        image_height_m=image_height_m,
        scale_m_per_px=scale_m_per_px,
        ds_factor=ds_factor,
        background_dir=background_dir,
        crop_roi=crop_roi,
        n_corrs_to_sum=n_corrs_to_sum,
        n_windows=n_windows,
        window_overlap=window_overlap,
        n_peaks=n_peaks,
        min_peak_dist_px=min_peak_dist_px,
        outlier_filter_mode=outlier_filter_mode,
        max_velocity_vy_vx_m_s=max_velocity_vy_vx_m_s,
        flow_direction=flow_direction,
        extra_vel_dim_m=extra_vel_dim_m,
        nb_size_tyx=nb_size_tyx,
        interp_nb_size_tyx=interp_nb_size_tyx,
        nb_threshold=nb_threshold,
        time_smooth_lam=time_smooth_lam,
        plot_model=plot_model,
        model_gender=model_gender,
        model_mass=model_mass,
        model_height=model_height,
        plot_global_filters=plot_global_filters,
        plot_correlations=plot_correlations,
        plot_window_layout=plot_window_layout,
        plot_flow_rate=plot_flow_rate,
        export_velocity_profiles_pdf=export_velocity_profiles_pdf,
    )


def _resolve_config_path(config_file: Path | str | None) -> Path:
    config_path = Path(config_file) if config_file else None
    if config_path and config_path.is_file():
        return config_path

    selected_path = ask_open_file(
        key="config_file",
        title="Select configuration file",
        filetypes=(("TOML files", "*.toml"), ("All files", "*.*")),
    )
    if selected_path is None:
        raise RuntimeError("No configuration file selected; aborting.")
    return Path(selected_path)


def _maybe_write_updated_config(
    *,
    config_path: Path,
    original_snapshot: dict[str, Any],
    image_dir: Path,
    output_dir: Path,
    frames_to_use: str | list[int],
    camera_dir: Path,
    calib_dir: Path,
    calib_spacing_mm: float,
    ds_factor: list[int],
    background_dir_for_config: str,
    background_dir: str,
    crop_roi: tuple[int, int, int, int],
) -> None:
    updated_snapshot = deepcopy(original_snapshot)
    updated_snapshot["source"].update(
        {
            "image_dir": str(image_dir),
            "output_dir": str(output_dir),
            "frames_to_use": frames_to_use,
        }
    )
    updated_snapshot["camera"].update(
        {
            "camera_dir": str(camera_dir),
            "calib_dir": str(calib_dir),
            "calib_spacing_mm": calib_spacing_mm,
        }
    )
    updated_snapshot["preprocessing"].update(
        {
            "ds_factor": ds_factor,
            "background_dir": (background_dir_for_config or background_dir),
            "crop_roi": list(crop_roi),
        }
    )

    changes = _diff_dicts(original_snapshot, updated_snapshot)
    if not changes:
        return

    print("Updated configuration values detected:")
    for key_path, old_value, new_value in changes:
        print(f" - {key_path}: {old_value!r} -> {new_value!r}")

    if not prompt_yes_no(
        f"Save updated configuration to {config_path}? A backup (.bak) will be created. [y/N]: "
    ):
        print("Skipping save. Changes not written to disk.")
        return

    backup_path = config_path.with_suffix(
        config_path.suffix + "_" + timestamp_str() + ".bak")
    shutil.copy2(config_path, backup_path)
    config_path.write_text(_toml_dump(updated_snapshot), encoding="utf-8")
    print(
        f"Saved updated configuration to {config_path}. Backup at {backup_path}.")


def _get_image_list(
    data_path: Path,
    frames_to_use: list[int] | str = "all",
    filetype: str = "tif",
    lead_zeros: int = 5,
) -> list[str]:
    """Helper function to get a list of image files from the data path."""
    image_files: list[str] = []

    if data_path.is_dir():
        all_files = natsorted(
            [f for f in os.listdir(data_path) if f.endswith(f".{filetype}")]
        )

        if frames_to_use == "all":
            image_files = [
                os.path.join(data_path, f) for f in all_files if not f.startswith(".")
            ]
        else:
            image_files = [
                os.path.join(data_path, f)
                for f in all_files
                if any(
                    f.endswith(f"{nr:0{lead_zeros}d}.{filetype}") for nr in frames_to_use
                )
                and not f.startswith(".")
            ]
    else:
        raise FileNotFoundError(
            f"Data path {data_path} is not a valid directory.")

    if not image_files:
        raise FileNotFoundError(f"No image files found in {data_path}.")

    return image_files


def _maybe_generate_background(
    *,
    image_paths: list[str],
    output_dir: Path,
    crop_roi: tuple[int, int, int, int],
    image_count: int,
) -> str:
    """Optionally generate, crop, and save a background image."""

    if image_count > 100:
        print(
            f"Warning: generating a background from {image_count} images may take some time.")

    if not prompt_yes_no("No background provided. Generate one now? [y/N]: "):
        print("Skipping background generation.")
        return ""

    print("Loading images to compute background...")
    imgs = load_images(image_paths, show_progress=True)
    background = generate_background(imgs)

    cropped_bg = crop(background, crop_roi)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"background_{timestamp_str()}.tif"
    tifffile.imwrite(output_path, cropped_bg)
    print(f"Background saved to {output_path}")
    return str(output_path)


def _read_packaged_default_config() -> dict[str, Any]:
    """Load packaged defaults from default_config.toml."""
    try:
        from importlib.resources import files

        default_path = files("tcm_piv").joinpath("config/default_config.toml")
        with default_path.open("rb") as fp:
            return tomllib.load(fp)
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Failed to load packaged default_config.toml. "
            "Make sure it is included as package data."
        ) from exc


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base, returning a new dict."""
    out = deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _toml_escape_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _toml_dump_value(value: Any) -> str:
    # TOML has no null; use empty string for optional paths.
    if value is None:
        return _toml_escape_string("")
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return _toml_escape_string(value)
    if isinstance(value, tuple):
        return "[" + ", ".join(_toml_dump_value(v) for v in value) + "]"
    if isinstance(value, list):
        return "[" + ", ".join(_toml_dump_value(v) for v in value) + "]"
    raise TypeError(f"Unsupported TOML value type: {type(value)!r}")


def _toml_dump(config: dict[str, Any]) -> str:
    """Serialize a limited TOML subset used by this project."""
    lines: list[str] = []

    for key, value in config.items():
        if not isinstance(value, dict):
            lines.append(f"{key} = {_toml_dump_value(value)}")

    for section, table in config.items():
        if not isinstance(table, dict):
            continue
        lines.append("")
        lines.append(f"[{section}]")
        for key, value in table.items():
            if isinstance(value, dict):
                raise TypeError(
                    "Nested tables beyond 1 level are not supported")
            lines.append(f"{key} = {_toml_dump_value(value)}")

    return "\n".join(lines).strip() + "\n"


def _is_scalar_sequence(value: Any) -> bool:
    return isinstance(value, list) and (len(value) == 0 or not isinstance(value[0], list))


def _as_int_tuple(value: Any, *, length: int) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"Expected an array of length {length}")
    try:
        return tuple(int(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected integer array of length {length}") from exc


def _as_float_tuple(value: Any, *, length: int) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise ValueError(f"Expected an array of length {length}")
    try:
        return tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected float array of length {length}") from exc


def _per_pass(
    value: Any,
    *,
    nr_passes: int,
    element_parser: Callable[[Any], Any] | None = None,
    tuple_len: int | None = None,
) -> list[Any]:
    """Normalize a parameter to a per-pass list of length nr_passes."""

    def parse_elem(v: Any) -> Any:
        return element_parser(v) if element_parser else v

    if (
        tuple_len is not None
        and isinstance(value, list)
        and _is_scalar_sequence(value)
        and len(value) == tuple_len
    ):
        elem = parse_elem(value)
        return [elem for _ in range(nr_passes)]

    if not isinstance(value, list):
        elem = parse_elem(value)
        return [elem for _ in range(nr_passes)]

    if len(value) == 1:
        elem = parse_elem(value[0])
        return [elem for _ in range(nr_passes)]

    if len(value) != nr_passes:
        raise ValueError(
            f"Expected list length 1 or {nr_passes}, got {len(value)}")
    return [parse_elem(v) for v in value]


def _diff_dicts(original: dict[str, Any], updated: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    """Return (key_path, old, new) for changes. Supports 1-level nested tables."""
    changes: list[tuple[str, Any, Any]] = []
    keys = set(original.keys()) | set(updated.keys())
    for key in sorted(keys):
        o = original.get(key)
        u = updated.get(key)
        if isinstance(o, dict) and isinstance(u, dict):
            subkeys = set(o.keys()) | set(u.keys())
            for sk in sorted(subkeys):
                ov = o.get(sk)
                uv = u.get(sk)
                if ov != uv:
                    changes.append((f"{key}.{sk}", ov, uv))
        else:
            if o != u:
                changes.append((key, o, u))
    return changes


if __name__ == "__main__":
    load_config(Path(__file__).resolve().parent / "config" / "config.toml")
