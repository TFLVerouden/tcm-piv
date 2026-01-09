"""Initializes all user configuration parameters. Provides function
`read_file()` to load a configuration file from disk.

Example usage for `main.py`:
    import load_config as cfg
    cfg.read_file("config.ini")

Namespace `cfg` now contains all user settings
"""

import os
import shutil
import configparser
from pathlib import Path
import sys

import numpy as np
import tifffile
from natsort import natsorted
from tcm_utils.camera_calibration import ensure_calibration
from tcm_utils.file_dialogs import ask_open_file
from tcm_utils.io_utils import ensure_path, load_images, prompt_yes_no
from tcm_utils.read_cihx import ensure_cihx_processed, load_cihx_metadata
from tcm_utils.time_utils import timestamp_str
from tcm_piv.preprocessing import crop, generate_background

# Default placeholder values for configuration parameters


# [Source]
FRAMES_TO_USE = 'all'

# [Camera]

# [Preprocessing]
DOWNSAMPLE_FACTOR = 1
BACKGROUND_DIR = ""
CROP_ROI = (0, 0, 0, 0)  # (y_start, y_end, x_start, x_end)

# [Correlation]

# [Peaks]

# [Postprocessing]

# [Modelling]

# [Visualisation]


def read_file(config_file: Path | str | None) -> None:
    """Reads a configuration file and sets module-level variables accordingly.

    Parameters:
        config_file (Path | str | None): Path to the configuration file.
    """
    config_path = Path(config_file) if config_file else None

    # If no config file provided, ask user to select one
    if not config_path or not config_path.is_file():
        print("No configuration file provided or file does not exist.")
        print("Please select a configuration file.")
        selected_path = ask_open_file(
            key="config_file",
            title="Select configuration file",
            filetypes=(("INI files", "*.ini"), ("All files", "*.*")),
        )
        if selected_path is None:
            print("WARNING: No configuration file selected. Using default parameters.")
            return

        config_path = Path(selected_path)

    print(f"Reading configuration from: {config_path}.")
    parser = configparser.ConfigParser()
    parser.read(config_path)
    original_snapshot = _snapshot_parser(parser)

    # [Source] =================================================================
    global IMAGE_DIR, OUTPUT_DIR, FRAMES_TO_USE, IMAGE_LIST, N_IMAGES
    cat = 'Source'

    # Get vars from config file or use defaults
    IMAGE_DIR = _get_cfg(parser, cat, 'image_dir')
    OUTPUT_DIR = _get_cfg(parser, cat, 'output_dir')
    FRAMES_TO_USE = _get_cfg(parser, cat, 'frames_to_use',
                             fallback=FRAMES_TO_USE)

    # Ensure paths are valid, prompt user if necessary
    IMAGE_DIR = ensure_path(IMAGE_DIR, 'image_dir',
                            title='Select image directory')
    OUTPUT_DIR = ensure_path(OUTPUT_DIR, 'output_dir',
                             title='Select output directory',
                             default_dir=Path(IMAGE_DIR))

    # Get the number of images by reading the image path
    IMAGE_LIST = _get_image_list(Path(IMAGE_DIR), FRAMES_TO_USE)
    N_IMAGES = len(IMAGE_LIST)
    if N_IMAGES < 2:
        raise Exception(f"Error: Found only {N_IMAGES} images in {IMAGE_DIR}.\n"
                        f"At least 2 images are required for PIV analysis.")
    print(f"Number of images to process: {N_IMAGES}")

    # [Camera] =================================================================
    global CAMERA_DIR, CALIB_DIR, CALIB_SPACING_MM, TIMESTAMP, FRAMERATE_HZ, SHUTTERSPEED_NS, RESOLUTION_X_PX, RESOLUTION_Y_PX
    cat = 'Camera'

    # TODO: test edge cases for "ensure*" functions
    CAMERA_DIR = _get_cfg(parser, cat, 'camera_dir')
    CALIB_DIR = _get_cfg(parser, cat, 'calib_dir')
    CALIB_SPACING_MM = _get_cfg(parser, cat, 'calib_spacing_mm')

    # Ensure that the camera metadata and calibration image are processed
    CAMERA_DIR = ensure_cihx_processed(
        CAMERA_DIR, output_dir=Path(OUTPUT_DIR + "/camera_metadata"))
    CALIB_DIR = ensure_calibration(CALIB_DIR, distance_mm=CALIB_SPACING_MM,
                                   output_dir=Path(OUTPUT_DIR + "/calibration"))

    # Load CIHX metadata
    cihx_metadata = load_cihx_metadata(Path(CAMERA_DIR))

    # Get some data from the CIHX metadata
    TIMESTAMP = cihx_metadata.get("timestamp")
    camera_meta = cihx_metadata.get("camera_metadata", {})
    FRAMERATE_HZ = camera_meta.get("recordRate")
    SHUTTERSPEED_NS = camera_meta.get("shutterSpeedNsec")
    RESOLUTION_X_PX = camera_meta.get(
        "resolution", {}).get("width")
    RESOLUTION_Y_PX = camera_meta.get(
        "resolution", {}).get("height")

    # [Preprocessing] ==========================================================
    global DOWNSAMPLE_FACTOR, BACKGROUND_DIR, CROP_ROI
    cat = 'Preprocessing'

    DOWNSAMPLE_FACTOR = int(
        _get_cfg(parser, cat, 'downsample_factor',
                 fallback=str(DOWNSAMPLE_FACTOR))
    )
    BACKGROUND_DIR = _get_cfg(parser, cat, 'background_dir')
    CROP_ROI = _parse_crop_roi(
        _get_cfg(parser, cat, 'crop_roi', fallback=_stringify(CROP_ROI)),
        default=CROP_ROI,
    )

    if not BACKGROUND_DIR:
        BACKGROUND_DIR = _maybe_generate_background(
            image_paths=IMAGE_LIST,
            output_dir=Path(OUTPUT_DIR),
            crop_roi=CROP_ROI,
            image_count=N_IMAGES,
        )

    # Additional sections can be added here following the same pattern

    # After reading all config values, check for changes
    updated_snapshot = _build_updated_snapshot(original_snapshot)
    changes = _diff_snapshots(original_snapshot, updated_snapshot)

    if not changes:
        print("No configuration values changed; nothing to save.")
        return
    print("Updated configuration values detected:")
    for section, option, old_value, new_value in changes:
        print(f" - [{section}] {option}: '{old_value}' -> '{new_value}'")

    # Prompt user to save changes
    save_prompt = input(
        f"Save updated configuration to {config_path}? A backup (.bak) will be created. [y/N]: "
    ).strip().lower()

    if save_prompt not in ("y", "yes"):
        print("Skipping save. Changes not written to disk.")
        return

    updated_parser = configparser.ConfigParser()
    updated_parser.read_dict(updated_snapshot)

    backup_path = config_path.with_suffix(
        config_path.suffix + "_" + timestamp_str() + ".bak")
    if config_path.is_file():
        shutil.copy2(config_path, backup_path)

    with config_path.open('w') as config_fp:
        updated_parser.write(config_fp)

    print(
        f"Saved updated configuration to {config_path}. Backup at {backup_path}.")


def _get_image_list(data_path: Path, frames_to_use: list[int] | str = 'all', filetype: str = 'tif', lead_zeros: int = 5) -> list[str]:
    """Helper function to get a list of image files from the data path.

    Args:
        data_path (Path): Path to the directory containing images.

    Returns:
        list[str]: List of image file paths.
    """
    image_files = []

    # List all files in the directory with the specified filetype
    if data_path.is_dir():
        all_files = natsorted(
            [f for f in os.listdir(data_path) if f.endswith(f".{filetype}")]
        )

        # Handle "all" option or specific frame numbers
        if frames_to_use == "all":
            # Load all images, exclude hidden files
            image_files = [
                os.path.join(data_path, f) for f in all_files
                if not f.startswith('.')
            ]
        else:
            # Filter files to include only those that match the specified frame numbers
            image_files = [
                os.path.join(data_path, f) for f in all_files
                if any(f.endswith(f"{nr:0{lead_zeros}d}.{filetype}") for nr in frames_to_use)
                and not f.startswith('.')
            ]
    else:
        raise FileNotFoundError(
            f"Data path {data_path} is not a valid directory.")

    if not image_files:
        raise FileNotFoundError(f"No image files found in {data_path}.")

    return image_files


def _check_variables() -> None:
    """Check that variables in the Correlation, Peaks, and Postprocessing categories all have the right length for the number of passes.

    Raises:
        ValueError: If any variable has an incorrect length."""
    pass  # Placeholder for future implementation


def _get_cfg(parser: configparser.ConfigParser, section: str, option: str,
             fallback: str = "") -> str:
    """Read a value from the config, treating empty strings as missing.

    Always returns a string; if the value is missing or empty, ``fallback``
    is returned instead.
    """

    try:
        if fallback == "":
            value = parser.get(section, option)
        else:
            value = parser.get(section, option, fallback=fallback)
    except (configparser.NoSectionError, configparser.NoOptionError):
        return fallback

    if isinstance(value, str) and value.strip() == "":
        return fallback
    return value


def _snapshot_parser(parser: configparser.ConfigParser) -> dict[str, dict[str, str]]:
    """Capture parser values for diffing."""
    snapshot: dict[str, dict[str, str]] = {}
    for section in parser.sections():
        snapshot[section] = {}
        for option in parser.options(section):
            snapshot[section][option] = parser.get(
                section, option, fallback="")
    return snapshot


def _stringify(value) -> str:
    """Convert values to strings for config storage."""
    if value is None:
        return ""
    if isinstance(value, (list, tuple, np.ndarray)):
        return ",".join(str(v) for v in value)
    return str(value)


def _set_value(parser: configparser.ConfigParser, section: str, option: str, value) -> None:
    """Set a value on the parser, creating the section if needed."""
    if not parser.has_section(section):
        parser.add_section(section)
    parser.set(section, option, _stringify(value))


def _diff_snapshots(original: dict[str, dict[str, str]], updated: dict[str, dict[str, str]]) -> list[tuple[str, str, str | None, str]]:
    """Return (section, option, old, new) for changed values."""
    changes: list[tuple[str, str, str | None, str]] = []
    sections = set(original.keys()) | set(updated.keys())
    for section in sections:
        orig_section = original.get(section, {})
        updated_section = updated.get(section, {})
        options = set(orig_section.keys()) | set(updated_section.keys())
        for option in options:
            old_value = orig_section.get(option)
            new_value = updated_section.get(option)
            if old_value != new_value:
                changes.append((section, option, old_value, new_value))
    return changes


def _build_updated_snapshot(original_snapshot: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    """Populate a new snapshot using current globals for any matching config keys."""
    updated_parser = configparser.ConfigParser()
    updated_parser.read_dict(original_snapshot)

    for section, options in original_snapshot.items():
        for option in options:
            global_name = option.upper()
            if global_name in globals():
                _set_value(updated_parser, section,
                           option, globals()[global_name])

    return _snapshot_parser(updated_parser)


def _parse_crop_roi(value, default: tuple[int, int, int, int] = (0, -1, 0, -1)) -> tuple[int, int, int, int]:
    """Parse crop ROI from config value into a 4-tuple of ints."""

    if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 4:
        try:
            return tuple(int(v) for v in value)  # type: ignore[return-value]
        except (TypeError, ValueError):
            return default

    if isinstance(value, str):
        cleaned = value.strip().strip("()[]")
        if cleaned:
            parts = [p.strip() for p in cleaned.split(',') if p.strip()]
            if len(parts) == 4:
                try:
                    # type: ignore[return-value]
                    return tuple(int(p) for p in parts)
                except ValueError:
                    return default

    return default


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

if __name__ == "__main__":
    # For testing purposes, read a sample config file
    read_file(Path("src/tcm_piv/empty_config.ini"))
