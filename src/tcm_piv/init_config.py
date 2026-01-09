"""Initializes all user configuration parameters. Provides function
`read_file()` to load a configuration file from disk.

Example usage for `main.py`:
    import load_config as cfg
    cfg.read_file("config.ini")

Namespace `cfg` now contains all user settings
"""

import os
import configparser
from pathlib import Path
import sys
from natsort import natsorted
import numpy as np
from tcm_utils.file_dialogs import ask_open_file
from tcm_utils.read_cihx import ensure_cihx_processed, extract_cihx_metadata, load_cihx_metadata
from tcm_utils.io_utils import ensure_path
from tcm_utils.camera_calibration import ensure_calibration

# Default placeholder values for configuration parameters
# FIXME

# [Source]
IMAGE_DIR = ""
CAMERA_DIR = ""
CALIB_DIR = ""
OUTPUT_DIR = ""
FRAMES_TO_USE = "all"

# [Preprocessing]
CALIB_SPACING_MM = 1.0

# [Correlation]

# [Peaks]

# [Postprocessing]

# [Modelling]

# [Visualisation]


def read_file(config_file: Path) -> None:
    """Reads a configuration file and sets module-level variables accordingly.

    Parameters:
        config_file (Path): Path to the configuration file.
    """
    # If no config file provided, ask user to select one
    if not config_file or not os.path.isfile(config_file):
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

        config_file = selected_path

    print(f"Reading configuration from: {config_file}.")
    parser = configparser.ConfigParser()
    parser.read(config_file)

    # [Source] =================================================================
    global IMAGE_DIR, OUTPUT_DIR, FRAMES_TO_USE, IMAGE_LIST, N_IMAGES
    category = 'Source'

    # Get vars from config file or use defaults
    IMAGE_DIR = parser.get(category, 'image_dir', fallback=IMAGE_DIR)
    OUTPUT_DIR = parser.get(category, 'output_dir', fallback=OUTPUT_DIR)
    FRAMES_TO_USE = parser.get(
        category, 'frames_to_use', fallback=FRAMES_TO_USE)

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
    category = 'Camera'

    # TODO: test edge cases for "ensure*" functions
    CAMERA_DIR = parser.get(category, 'camera_dir', fallback=CAMERA_DIR)
    CALIB_DIR = parser.get(category, 'calib_dir', fallback=CALIB_DIR)
    CALIB_SPACING_MM = parser.getfloat(
        category, 'calib_spacing_mm', fallback=CALIB_SPACING_MM)

    # Ensure that the camera metadata and calibration image are processed
    CAMERA_DIR = ensure_cihx_processed(CAMERA_DIR, output_dir=Path(OUTPUT_DIR))
    CALIB_DIR = ensure_calibration(CALIB_DIR, distance_mm=CALIB_SPACING_MM,
                                   output_dir=Path(OUTPUT_DIR))

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
    global DOWNSAMPLE_FACTOR
    category = 'Preprocessing'

    DOWNSAMPLE_FACTOR = parser.getint(
        category, 'downsample_factor', fallback=1)

    # Additional sections can be added here following the same pattern

    # Finally, ask if user wants to save updated config with paths to processed
    # camera metadata, calibration image and, if applicable, background image


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


if __name__ == "__main__":
    # For testing purposes, read a sample config file
    read_file(Path("src/tcm_piv/config.ini"))
