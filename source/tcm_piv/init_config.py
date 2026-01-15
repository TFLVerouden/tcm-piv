"""Configuration loading for tcm-piv.

This module provides :func:`read_file` which loads a TOML configuration,
applies packaged defaults, normalizes pass-dependent parameters, and
exposes the resulting settings as module-level variables."""

from __future__ import annotations

import json
import os
import shutil
import tomllib
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import tifffile
from natsort import natsorted
from tcm_utils.camera_calibration import ensure_calibration
from tcm_utils.file_dialogs import ask_open_file
from tcm_utils.io_utils import ensure_path, load_images, load_metadata, prompt_yes_no
from tcm_utils.read_cihx import ensure_cihx_processed
from tcm_utils.time_utils import timestamp_str
from tcm_piv.preprocessing import crop, generate_background


# These variables are populated by read_file(). Defaults live in default_config.toml.
#
# MAINTENANCE GUIDE (adding/changing config variables)
# --------------------------------------------------
# The config system is intentionally “TOML-driven”:
# - `default_config.toml` is the single source of truth for defaults.
# - `read_file()` loads user TOML, deep-merges it over the defaults, and then
#   normalizes/types values before exposing them as module-level variables.
# - The Python code should NOT hard-code fallback defaults in `.get(..., default)`.
#
# When you ADD a new config variable:
# 1) Add it to `default_config.toml` under the right table.
#    - Every key that `read_file()` indexes (e.g. `preprocessing["crop_roi"]`)
#      must exist in defaults.
#    - Keep tables only 1 level deep (this project’s TOML writer `_toml_dump()`
#      does not support nested tables beyond `[section] key = value`).
#    - TOML has no `null`. For “optional path” style values, use the convention
#      of an empty string `""` meaning “not provided”.
#
# 2) Add/update the example config (`config.toml`) so users see the new option.
#    - Put a realistic example value and short comment.
#
# 3) Add a module-level type annotation here.
#    - This is for IDE/type-checking and for documenting what `read_file()` sets.
#
# 4) In `read_file()`, read the value from the merged config and normalize it.
#    Typical patterns:
#    - Scalar (single value): `value = float(table["my_key"])`
#    - Per-pass: use `_normalize_per_pass(...)`.
#      * If it can be either scalar or per-pass list, `_normalize_per_pass` will
#        broadcast a scalar or `[x]` to length `nr_passes`.
#      * For “tuple-like” arrays (e.g. `[w, h]`), use `_as_int_tuple/_as_float_tuple`
#        and pass `tuple_len=...` so `[w, h]` is treated as one element, not as
#        per-pass list.
#
# 5) If runtime code can “fill in” this value (e.g. prompting for paths,
#    generating a background, running ensure_*), decide whether it should be
#    persisted back to the user config.
#    - If yes: add it to `updated_snapshot[...]` so it appears in the diff and
#      can be saved (with backup) to the original TOML.
#    - If no: keep it as a runtime-only variable.
#
# When you CHANGE a variable (rename/change type/change meaning):
# - Update `default_config.toml`, `config.toml`, and the corresponding parsing
#   logic in `read_file()` together.
# - If you rename keys, consider temporarily supporting the old key by mapping
#   it in code during a transition period (but keep defaults TOML consistent).
#
# After changes:
# - Run `python -m tcm_piv.main path/to/config.toml` (or your normal entrypoint)
#   to sanity-check parsing and the “update config with backup” flow.
NR_PASSES: int

# TODO: Add rectangle filter options?

# [source]
FRAMES_TO_USE: str | list[int]

# [preprocessing]
DOWNSAMPLE_FACTOR: list[int] | int
BACKGROUND_DIR: str
CROP_ROI: tuple[int, int, int, int]  # (y_start, y_end, x_start, x_end)

# [correlation]
CORRS_TO_SUM: list[int] | int
NR_WINDOWS: list[tuple[int, int]] | tuple[int, int]
WINDOW_OVERLAP: list[float] | float

# [displacement]
NR_PEAKS: list[int] | int
MIN_PEAK_DISTANCE: list[int] | int

# [postprocessing]
MAX_VELOCITY: list[tuple[float, float]] | tuple[float, float]
NEIGHBOURHOOD_SIZE: list[tuple[int, int, int]] | tuple[int, int, int]
NEIGHBOURHOOD_THRESHOLD: list[int | tuple[int, int]] | int | tuple[int, int]
TIME_SMOOTHING_LAMBDA: list[float] | float
FLOW_DIRECTION: str
EXTRA_VEL_DIM_M: float
OUTLIER_FILTER_MODE: list[str] | str


# [visualisation]
PLOT_MODEL: bool
MODEL_GENDER: str
MODEL_MASS: float
MODEL_HEIGHT: float
PLOT_GLOBAL_FILTERS: bool
WINDOW_PLOT_ENABLED: list[bool]
PLOT_CORRELATIONS: bool
PLOT_FLOW_RATE: bool
EXPORT_VELOCITY_PROFILES_PDF: bool


def read_file(config_file: Path | str | None) -> None:
    """Load TOML config, apply defaults, normalize settings, and set globals.

    If no valid config file is provided, the user is prompted to select one.
    Canceling the prompt aborts.
    """

    config_path = Path(config_file) if config_file else None
    if not config_path or not config_path.is_file():
        selected_path = ask_open_file(
            key="config_file",
            title="Select configuration file",
            filetypes=(("TOML files", "*.toml"), ("All files", "*.*")),
        )
        if selected_path is None:
            raise RuntimeError("No configuration file selected; aborting.")
        config_path = Path(selected_path)

    print(f"Reading configuration from: {config_path}.")
    with config_path.open("rb") as fp:
        user_cfg = tomllib.load(fp)

    defaults = _read_packaged_default_config()
    merged = _deep_merge(defaults, user_cfg)
    original_snapshot = deepcopy(merged)

    global NR_PASSES
    NR_PASSES = int(merged["nr_passes"])
    if NR_PASSES < 1:
        raise ValueError("nr_passes must be >= 1")

    source = merged["source"]
    camera = merged["camera"]
    preprocessing = merged["preprocessing"]
    correlation = merged["correlation"]
    displacement = merged["displacement"]
    postprocessing = merged["postprocessing"]
    visualisation = merged["visualisation"]

    # [source]
    global IMAGE_DIR, OUTPUT_DIR, FRAMES_TO_USE, IMAGE_LIST, NR_IMAGES
    IMAGE_DIR = str(source["image_dir"])
    OUTPUT_DIR = str(source["output_dir"])
    FRAMES_TO_USE = source["frames_to_use"]
    if not (FRAMES_TO_USE == "all" or isinstance(FRAMES_TO_USE, list)):
        raise ValueError(
            "source.frames_to_use must be 'all' or an array of integers")
    if isinstance(FRAMES_TO_USE, list):
        FRAMES_TO_USE = [int(v) for v in FRAMES_TO_USE]

    IMAGE_DIR = str(ensure_path(IMAGE_DIR, "image_dir",
                    title="Select image directory"))
    OUTPUT_DIR = str(ensure_path(
        OUTPUT_DIR,
        "output_dir",
        title="Select output directory",
        default_dir=Path(IMAGE_DIR),
    ))

    IMAGE_LIST = _get_image_list(Path(IMAGE_DIR), FRAMES_TO_USE)
    NR_IMAGES = len(IMAGE_LIST)
    if NR_IMAGES < 2:
        raise RuntimeError(
            f"Error: Found only {NR_IMAGES} images in {IMAGE_DIR}. "
            "At least 2 images are required for PIV analysis."
        )
    print(f"Number of images to process: {NR_IMAGES}")

    # [camera]
    global CAMERA_DIR, CALIB_DIR, CALIB_SPACING_MM, TIMESTAMP, FRAMERATE_HZ, TIMESTEP_S, SHUTTERSPEED_NS, IMAGE_WIDTH_PX, IMAGE_HEIGHT_PX, IMAGE_WIDTH_M, IMAGE_HEIGHT_M, SCALE_M_PER_PX
    CAMERA_DIR = str(camera["camera_dir"])
    CALIB_DIR = str(camera["calib_dir"])
    CALIB_SPACING_MM = float(camera["calib_spacing_mm"])

    # TODO: test edge cases for "ensure*" functions

    CAMERA_DIR = str(
        ensure_cihx_processed(
            Path(CAMERA_DIR) if CAMERA_DIR else None,
            output_dir=Path(OUTPUT_DIR) / "camera_metadata",
        )
    )
    CALIB_DIR = str(
        ensure_calibration(
            Path(CALIB_DIR) if CALIB_DIR else None,
            distance_mm=CALIB_SPACING_MM,
            output_dir=Path(OUTPUT_DIR) / "calibration",
        )
    )

    camera_metadata = load_metadata(Path(CAMERA_DIR))
    TIMESTAMP = str(camera_metadata.get("timestamp"))
    camera_meta = camera_metadata.get("camera_metadata", {})
    FRAMERATE_HZ = int(camera_meta.get("recordRate"))
    TIMESTEP_S = 1.0 / FRAMERATE_HZ  # Formerly called "dt"
    SHUTTERSPEED_NS = int(camera_meta.get("shutterSpeedNsec"))
    IMAGE_WIDTH_PX = int(camera_meta.get("resolution", {}).get("width"))
    IMAGE_HEIGHT_PX = int(camera_meta.get("resolution", {}).get("height"))

    calib_metadata = load_metadata(Path(CALIB_DIR))
    IMAGE_WIDTH_M = float(calib_metadata.get(
        "calibration", {}).get("image_size_m", {}).get("width"))
    IMAGE_HEIGHT_M = float(calib_metadata.get(
        # formerly called "frame_w" (as the images are rotated...)
        "calibration", {}).get("image_size_m", {}).get("height"))
    SCALE_M_PER_PX = float(calib_metadata.get(
        "calibration", {}).get("scale_m_per_px"))

    # [preprocessing]
    global DOWNSAMPLE_FACTOR, BACKGROUND_DIR, CROP_ROI
    DOWNSAMPLE_FACTOR = _per_pass(
        preprocessing["downsample_factor"],
        nr_passes=NR_PASSES,
        element_parser=lambda v: int(v),
    )
    BACKGROUND_DIR = preprocessing["background_dir"]
    BACKGROUND_DIR = "" if BACKGROUND_DIR is None else str(BACKGROUND_DIR)

    roi_value = preprocessing["crop_roi"]
    CROP_ROI = tuple(_as_int_tuple(roi_value, length=4)
                     )  # type: ignore[assignment]

    if not BACKGROUND_DIR:
        BACKGROUND_DIR = _maybe_generate_background(
            image_paths=IMAGE_LIST,
            output_dir=Path(OUTPUT_DIR),
            crop_roi=CROP_ROI,
            image_count=NR_IMAGES,
        )

    # [correlation]
    global CORRS_TO_SUM, NR_WINDOWS, WINDOW_OVERLAP
    CORRS_TO_SUM = _per_pass(
        correlation["corrs_to_sum"],
        nr_passes=NR_PASSES,
        element_parser=lambda v: int(v),
    )
    NR_WINDOWS = _per_pass(
        correlation["nr_windows"],
        nr_passes=NR_PASSES,
        element_parser=lambda v: _as_int_tuple(v, length=2),
        tuple_len=2,
    )
    WINDOW_OVERLAP = _per_pass(
        correlation["window_overlap"],
        nr_passes=NR_PASSES,
        element_parser=lambda v: float(v),
    )

    # [displacement]
    global NR_PEAKS, MIN_PEAK_DISTANCE
    NR_PEAKS = _per_pass(
        displacement["nr_peaks"],
        nr_passes=NR_PASSES,
        element_parser=lambda v: int(v),
    )
    MIN_PEAK_DISTANCE = _per_pass(
        displacement["min_peak_distance"],
        nr_passes=NR_PASSES,
        element_parser=lambda v: int(v),
    )

    # [postprocessing]
    global MAX_VELOCITY, NEIGHBOURHOOD_SIZE, NEIGHBOURHOOD_THRESHOLD, TIME_SMOOTHING_LAMBDA, FLOW_DIRECTION, EXTRA_VEL_DIM_M, OUTLIER_FILTER_MODE
    MAX_VELOCITY = _per_pass(
        postprocessing["max_velocity"],
        nr_passes=NR_PASSES,
        element_parser=lambda v: _as_float_tuple(v, length=2),
        tuple_len=2,
    )
    NEIGHBOURHOOD_SIZE = _per_pass(
        postprocessing["neighbourhood_size"],
        nr_passes=NR_PASSES,
        element_parser=lambda v: _as_int_tuple(v, length=3),
        tuple_len=3,
    )

    def _parse_threshold(v: Any) -> int | tuple[int, int]:
        if isinstance(v, list):
            if len(v) != 2:
                raise ValueError(
                    "neighbourhood_threshold tuple must have length 2")
            return (int(v[0]), int(v[1]))
        return int(v)

    thr_value = postprocessing["neighbourhood_threshold"]
    if (
        isinstance(thr_value, list)
        and _is_scalar_sequence(thr_value)
        and len(thr_value) == 2
        and all(isinstance(x, (int, float)) for x in thr_value)
    ):
        NEIGHBOURHOOD_THRESHOLD = [_parse_threshold(
            thr_value) for _ in range(NR_PASSES)]
    else:
        NEIGHBOURHOOD_THRESHOLD = _per_pass(
            thr_value,
            nr_passes=NR_PASSES,
            element_parser=_parse_threshold,
        )

    TIME_SMOOTHING_LAMBDA = _per_pass(
        postprocessing["time_smoothing_lambda"],
        nr_passes=NR_PASSES,
        element_parser=lambda v: float(v),
    )

    OUTLIER_FILTER_MODE = _per_pass(
        postprocessing["outlier_filter_mode"],
        nr_passes=NR_PASSES,
        element_parser=lambda v: str(v).strip().lower(),
    )
    allowed_outlier_modes = {"semicircle_rect", "circle"}
    for mode in OUTLIER_FILTER_MODE:
        if mode not in allowed_outlier_modes:
            raise ValueError(
                "postprocessing.outlier_filter_mode must be 'semicircle_rect' or 'circle' (scalar or per-pass array)"
            )

    FLOW_DIRECTION = str(postprocessing["flow_direction"]).strip().lower()
    if FLOW_DIRECTION not in {"x", "y"}:
        raise ValueError("postprocessing.flow_direction must be 'x' or 'y'")

    # formerly called "depth"
    EXTRA_VEL_DIM_M = float(postprocessing["extra_vel_dim_m"])

    # [visualisation]
    global PLOT_MODEL, MODEL_GENDER, MODEL_MASS, MODEL_HEIGHT
    global PLOT_GLOBAL_FILTERS, WINDOW_PLOT_ENABLED, PLOT_CORRELATIONS, PLOT_FLOW_RATE, EXPORT_VELOCITY_PROFILES_PDF
    PLOT_MODEL = bool(visualisation["plot_model"])
    MODEL_GENDER = str(visualisation["model_gender"])
    MODEL_MASS = float(visualisation["model_mass"])
    MODEL_HEIGHT = float(visualisation["model_height"])

    PLOT_GLOBAL_FILTERS = bool(visualisation["plot_global_filters"])
    PLOT_CORRELATIONS = bool(visualisation["plot_correlations"])
    PLOT_FLOW_RATE = bool(visualisation["plot_flow_rate"])
    EXPORT_VELOCITY_PROFILES_PDF = bool(
        visualisation["export_velocity_profiles_pdf"]
    )
    WINDOW_PLOT_ENABLED = _per_pass(
        visualisation["plot_window_layout"],
        nr_passes=NR_PASSES,
        element_parser=lambda v: bool(v),
    )

    # Offer to update config if runtime adjustments changed values
    updated_snapshot = deepcopy(original_snapshot)
    updated_snapshot["nr_passes"] = NR_PASSES

    updated_snapshot.setdefault("source", {})
    updated_snapshot["source"].update(
        {
            "image_dir": IMAGE_DIR,
            "output_dir": OUTPUT_DIR,
            "frames_to_use": FRAMES_TO_USE,
        }
    )
    updated_snapshot.setdefault("camera", {})
    updated_snapshot["camera"].update(
        {
            "camera_dir": CAMERA_DIR,
            "calib_dir": CALIB_DIR,
            "calib_spacing_mm": CALIB_SPACING_MM,
        }
    )
    updated_snapshot.setdefault("preprocessing", {})
    updated_snapshot["preprocessing"].update(
        {
            "downsample_factor": DOWNSAMPLE_FACTOR,
            "background_dir": BACKGROUND_DIR,
            "crop_roi": list(CROP_ROI),
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

    try:
        cropped_bg = crop(background, crop_roi)
    except ValueError as exc:
        print(
            f"Invalid crop ROI {crop_roi}; saving uncropped background. ({exc})")
        cropped_bg = background

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
    read_file(Path(__file__).resolve().parent / "config" / "config.toml")
