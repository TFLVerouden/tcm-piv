"""
Input/output functions for PIV analysis.

This module handles file operations including reading images,
loading/saving backup data, and saving figures.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from natsort import natsorted
from tqdm import tqdm


def save_backup(proc_path: str, file_name: str,
                test_mode=False, **kwargs) -> bool:
    """
    Save a backup file to the specified path.

    Args:
        proc_path (str): Path to the directory containing the backup file.
        filename (str): Name of the backup file to load/save.
        test_mode (bool, optional): If True, do nothing.
        **kwargs: Variables to save. Use as: backup(path, file, var1=value1, var2=value2, ...)

    Returns:
        success (bool): Whether the operation was successful.
    """
    # Skip if in test mode
    if test_mode:
        return False

    # Check for empty kwargs
    if not kwargs:
        print("Warning: No variables provided for saving.")
        return False

    # Check whether the supplied file name already has an extension
    if not file_name.endswith('.npz'):
        file_name += '.npz'

    # Save the variables to a .npz file
    file_path = os.path.join(proc_path, file_name)
    np.savez(file_path, **kwargs)
    print(f"Saved data to {file_path}")
    return True


def init_subfolder(*path_components, debug=False, verbose=True) -> str:
    """
    Create a folder path from multiple components if it doesn't exist.
    TODO: Change all test_mode to debug for consistency.
    Args:
        *path_components: Variable number of path components to join together.
        debug (bool, optional): If True, skip creation and just return the path.
        verbose (bool, optional): If True, print creation message.

    Returns:
        folder_path (str): Full path to the created folder.

    Examples:
        create_subfolder('/base/path', 'subfolder')  # Creates /base/path/subfolder
        create_subfolder('/base', 'series', 'name', 'data')  # Creates /base/series/name/data
    """
    if not path_components:
        raise ValueError("At least one path component must be provided")

    folder_path = os.path.join(*path_components)

    if not os.path.exists(folder_path) and not debug:
        os.makedirs(folder_path)
        if verbose:
            print(f"Created directory: {folder_path}")

    return folder_path


def load_backup(proc_path: str, file_name: str, var_names=None,
                test_mode=False) -> dict:
    """
    Load a backup file from the specified path.

    Args:
        proc_path (str): Path to the directory containing the backup file.
        filename (str): Name of the backup file to load/save.
        var_names (list, optional): List of variable names to load. If None, loads all variables.
        test_mode (bool, optional): If True, do nothing.

    Returns:
        loaded_vars (dict): Empty if nothing was loaded.
    """
    # Skip if in test mode
    if test_mode:
        return {}

    # Check whether the supplied file name already has an extension
    if not file_name.endswith('.npz'):
        file_name += '.npz'
    file_path = os.path.join(proc_path, file_name)

    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Warning: backup file {file_path} not found.")
        return {}

    # Load the data from the .npz file
    loaded_vars = {}
    with np.load(file_path) as data:
        if var_names is None:
            # Load all variables in the file
            for k in data.files:
                loaded_vars[k] = data[k]
        else:
            # Load only requested variables
            for k in var_names:
                if k in data:
                    loaded_vars[k] = data[k]
                else:
                    print(f"Warning: {k} not found in {file_path}")
    print(f"Loaded data from {file_path}")
    return loaded_vars

# TODO: Look for use of these load image(s) functions and replace by new ones from tcm-utils


def read_img(file_path: str) -> np.ndarray | None:
    """
    Read a single image file using OpenCV.

    Args:
        filepath (str): Full path to the image file to load.

    Returns:
        np.ndarray | None: Loaded image as grayscale array, or None if loading failed.
    """
    img = cv.imread(file_path, cv.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Warning: Failed to load {file_path}")
    return img


def read_imgs(data_path: str, frame_nrs: list[int] | str, format: str = 'tif', lead_0: int = 5, only_count: bool = False, timing: bool = True) -> np.ndarray:
    """
    Load selected images from a directory into a 3D numpy array.

    Args:
        data_path (str): Path to the directory containing images.
        frame_nrs (list[int] | str): List of frame numbers to load,
            or "all" to load all images.
        format (str): File extension to load.
        lead_0 (int): Number of leading zeros in the file names.
        only_count (bool): If True, only count the number of images without loading them.
        timing (bool): If True, show a progress bar while loading images.

    Returns:
        np.ndarray | int: 3D array of images (image_index, y, x) or the number of images found.
    """

    # Check if the directory exists
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data directory not found: {data_path}")

    # List all files in the directory
    files = natsorted(
        [f for f in os.listdir(data_path) if f.endswith('.' + format)])

    # Handle "all" option or specific frame numbers
    if frame_nrs == "all":
        # Load all images - filter by format and exclude hidden files
        files = [f for f in files if f.endswith(
            '.' + format) and not f.startswith('.')]
    else:
        # Filter files to include only those that match the specified frame numbers
        files = [f for f in files if any(f.endswith(f"{nr:0{lead_0}d}.{format}") for nr
                                         in frame_nrs) and not f.startswith('.')]

    if not files:
        raise FileNotFoundError(
            f"No files found in {data_path} with the specified criteria and format '{format}'.")

    if only_count:
        return len(files)

    # Read images into a 3D numpy array in parallel
    file_paths = [os.path.join(data_path, f) for f in files]

    n_jobs = os.cpu_count() or 4

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        imgs = list(tqdm(executor.map(read_img, file_paths),
                         total=len(file_paths),
                         desc='Reading images      '))

    # Convert list of images to a numpy array
    imgs = np.array(imgs, dtype=np.uint64)
    return imgs


def save_cfig(directory: str, file_name: str, format: str = 'pdf', test_mode: bool = False, verbose: bool = True):
    """
    Save the current matplotlib figure to a file.

    Args:
        directory (str): Directory to save the figure.
        filename (str): Name of the file to save the figure as.
        format (str): File format to save the figure in (e.g., 'pdf', 'png').
        test_mode (bool): If True, do not save the figure.
        verbose (bool): If True, print a message when saving the figure.

    Returns:
        None
    """

    # Only run when not in test mode
    if test_mode:
        return

    # Otherwise, save figure
    else:
        # Set directory and file format
        file_name = f"{file_name}.{format}"
        filepath = os.path.join(directory, file_name)

        # Save the figure
        plt.savefig(filepath, transparent=True, bbox_inches='tight',
                    format=format)
        if verbose:
            print(f"Figure saved to {filepath}")

    # # Show the figure
    # plt.show()

    return
