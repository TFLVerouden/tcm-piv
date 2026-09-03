"""Configuration loading for tcm-piv.

This module provides :func:`load_config`, which:
- loads a user TOML config,
- deep-merges it over packaged defaults,
- broadcasts per-pass parameters,
- resolves required input files (camera + calibration),
- and returns a single :class:`Config` object.

`Config` is a thin attribute-access wrapper over the merged/resolved dict:
any key from `default_config.toml` (optionally overridden by the user) is
available as `config.<key>`. Adding a new plain (non-per-pass) field only
requires adding it to `default_config.toml` - no code changes needed here.
Type/range validation of individual field values is left to whatever code
consumes them, not this module.

The end-to-end pipeline in :mod:`tcm_piv.run` is the primary consumer.
"""

from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
import shutil
import tomllib
from typing import Any

import tifffile
from natsort import natsorted

from tcm_piv.preprocessing import crop, get_roi_size, generate_background
from tcm_utils.camera_calibration import ensure_calibration
from tcm_utils.file_dialogs import ask_open_file
from tcm_utils.io_utils import ensure_path, load_images, load_metadata, prompt_yes_no
from tcm_utils.read_cihx import ensure_cihx_processed
from tcm_utils.time_utils import timestamp_str

# Per-pass fields must be given as a TOML array of length 1 (broadcast to
# every pass) or exactly `nr_passes` (one entry per pass). These sections are
# per-pass by default; a few fields are exceptions in either direction.
PER_PASS_SECTIONS = {"correlation", "displacement", "postprocessing"}
PER_PASS_KEYS = {"ds_factor", "plot_window_layout"}
NON_PER_PASS_KEYS = {"flow_direction", "extra_vel_dim_m"}


class Config:
    """Attribute-access wrapper around the merged/resolved config dict.

    Any key present in `default_config.toml` (or overridden by the user's
    TOML, or added during resolution) is exposed as `config.<key>`, regardless
    of which `[section]` it lives in. Per-pass fields are lists indexed by
    `pass_idx0`.
    """

    def __init__(self, data: dict[str, Any], *, config_path: Path):
        self._data = data
        self.config_path = config_path

    def __getattr__(self, name: str) -> Any:
        data = self._data
        if name in data and not isinstance(data[name], dict):
            return data[name]
        for value in data.values():
            if isinstance(value, dict) and name in value:
                return value[name]
        raise AttributeError(f"Config has no field {name!r}")


def load_config(
    config_file: Path | str | None,
    *,
    overrides: dict[str, Any] | None = None,
) -> tuple[Config, Path]:
    """Load TOML config, merge with packaged defaults, and resolve it."""

    config_path = _resolve_config_path(config_file)
    print(f"Reading configuration from: {config_path}.")

    with config_path.open("rb") as fp:
        user_cfg = tomllib.load(fp)

    merged = _deep_merge(_read_packaged_default_config(), user_cfg)
    if overrides:
        merged = _deep_merge(merged, overrides)

    nr_passes = int(merged["nr_passes"])
    if nr_passes < 1:
        raise ValueError("nr_passes must be >= 1")

    _broadcast_per_pass(merged, nr_passes=nr_passes)
    _resolve(merged)

    return Config(merged, config_path=config_path), config_path


def archive_config(config_path: Path, output_dir: Path) -> None:
    """Copy the used config into the run's output_dir for provenance."""
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / \
        f"{config_path.stem}_{timestamp_str()}{config_path.suffix}"
    shutil.copy2(config_path, archive_path)
    print(f"Archived a copy of the used configuration to {archive_path}.")


def _broadcast_per_pass(merged: dict[str, Any], *, nr_passes: int) -> None:
    """Expand per-pass fields to length `nr_passes`, in place."""

    for section, fields in merged.items():
        if not isinstance(fields, dict):
            continue
        for key, value in fields.items():
            is_per_pass = key in PER_PASS_KEYS or (
                section in PER_PASS_SECTIONS and key not in NON_PER_PASS_KEYS
            )
            if not is_per_pass:
                continue
            if not isinstance(value, list):
                raise ValueError(
                    f"{section}.{key} is a per-pass field; wrap it in an array, e.g. [{value!r}]"
                )
            if len(value) == 1:
                fields[key] = [deepcopy(value[0]) for _ in range(nr_passes)]
            elif len(value) != nr_passes:
                raise ValueError(
                    f"{section}.{key}: expected an array of length 1 or nr_passes ({nr_passes}), "
                    f"got {len(value)}"
                )


def _resolve(merged: dict[str, Any]) -> None:
    """Resolve file paths, camera/calibration metadata, and derived sizes.

    Mutates `merged` in place, adding/overwriting keys with concrete values.
    """

    source = merged["source"]
    camera = merged["camera"]
    preprocessing = merged["preprocessing"]

    # --- source ---
    frames_to_use = source["frames_to_use"]
    if isinstance(frames_to_use, list) and len(frames_to_use) == 2:
        start, end = frames_to_use
        frames_to_use = list(range(int(start), int(end) + 1))
        source["frames_to_use"] = frames_to_use

    image_dir = Path(ensure_path(
        str(source["image_dir"]), "image_dir", title="Select image directory"))
    output_dir = Path(ensure_path(
        str(source["output_dir"]),
        "output_dir",
        title="Select output directory",
        default_dir=image_dir,
    ))
    source["image_dir"] = image_dir
    source["output_dir"] = output_dir

    image_list = _get_image_list(image_dir, frames_to_use)
    if len(image_list) < 2:
        raise RuntimeError(
            f"Found only {len(image_list)} images in {image_dir}. At least 2 images are required."
        )
    source["image_list"] = image_list
    source["nr_images"] = len(image_list)

    # --- camera + calibration ---
    camera_dir = Path(
        ensure_cihx_processed(
            _optional_path(camera["camera_dir"]),
            output_dir=output_dir / "camera_metadata",
        )
    )
    calib_dir = Path(
        ensure_calibration(
            _optional_path(camera["calib_dir"]),
            distance_mm=float(camera["calib_spacing_mm"]),
            output_dir=output_dir / "calibration",
        )
    )
    camera["camera_dir"] = camera_dir
    camera["calib_dir"] = calib_dir

    camera_metadata = load_metadata(camera_dir)
    camera_meta = camera_metadata.get("camera_metadata", {})
    camera["timestamp"] = camera_metadata.get("timestamp")
    framerate_hz = int(camera_meta.get("recordRate"))
    camera["framerate_hz"] = framerate_hz
    camera["timestep_s"] = 1.0 / framerate_hz
    camera["shutterspeed_ns"] = camera_meta.get("shutterSpeedNsec")
    image_width_px = int(camera_meta.get("resolution", {}).get("width"))
    image_height_px = int(camera_meta.get("resolution", {}).get("height"))

    calib_metadata = load_metadata(calib_dir)
    scale_m_per_px = calib_metadata.get(
        "calibration", {}).get("scale_m_per_px")
    camera["scale_m_per_px"] = scale_m_per_px

    # The rest of the pipeline works in cropped image dimensions.
    image_height_px_crop, image_width_px_crop = get_roi_size(
        (image_height_px, image_width_px), preprocessing["crop_roi"]
    )
    camera["image_width_px"] = image_width_px_crop
    camera["image_height_px"] = image_height_px_crop
    camera["image_width_m"] = image_width_px_crop * scale_m_per_px
    camera["image_height_m"] = image_height_px_crop * scale_m_per_px

    # --- preprocessing ---
    bg_raw = str(preprocessing["background_dir"] or "").strip()
    if bg_raw.lower() in {"none", "null"}:
        preprocessing["background_dir"] = ""
    elif not bg_raw:
        preprocessing["background_dir"] = _maybe_generate_background(
            image_paths=image_list,
            output_dir=output_dir,
            crop_roi=preprocessing["crop_roi"],
        )
    # else: an explicit path was given, keep it as-is.


def _optional_path(value: Any) -> Path | None:
    s = "" if value is None else str(value).strip()
    return Path(s) if s else None


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
) -> str:
    """Optionally generate, crop, and save a background image."""

    if len(image_paths) > 100:
        print(
            f"Warning: generating a background from {len(image_paths)} images may take some time.")

    if not prompt_yes_no("No background provided. Generate one now? [y/N]: "):
        print("Skipping background generation.")
        return ""

    print("Loading images to compute background...")
    imgs = load_images(image_paths, show_progress=True)
    print("Generating background image...")
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


if __name__ == "__main__":
    load_config(Path(__file__).resolve().parent / "config" / "config.toml")
