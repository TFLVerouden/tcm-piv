import os
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from natsort import natsorted
from scipy import signal as sig
from scipy.interpolate import make_smoothing_spline
from skimage.feature import peak_local_max
from tqdm import trange, tqdm
from matplotlib import animation as ani

# Add the functions directory to the path and import CVD check
sys.path.append(os.path.join(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))), 'functions'))
import cvd_check as cvd

# TODO: Set up module.
""""
piv_functions/
├── __init__.py
├── correlation.py
├── filtering.py
├── io.py
├── plotting.py
├── processing.py
└── utils.py

E.g.:
# Import all functions from submodules
from .io import backup, read_img, read_imgs
from .processing import downsample, split_n_shift
from .correlation import calc_corr, calc_corrs, sum_corr, sum_corrs
from .utils import find_peaks, three_point_gauss, subpixel, find_disp, find_disps
from .filtering import filter_outliers, filter_neighbours, cart2polar, validate_n_nbs, first_valid, strip_peaks
from .plotting import save_cfig
from .processing import smooth

# Optional: define __all__ for explicit exports
__all__ = [
    'backup', 'read_img', 'read_imgs',
    'downsample', 'split_n_shift', 
    'calc_corr', 'calc_corrs', 'sum_corr', 'sum_corrs',
    'find_peaks', 'three_point_gauss', 'subpixel', 'find_disp', 'find_disps',
    'filter_outliers', 'filter_neighbours', 'cart2polar', 'validate_n_nbs', 
    'first_valid', 'strip_peaks', 'smooth', 'save_cfig'
]

"""

def get_time(frames: list[int], dt: float) -> np.ndarray:
    """
    Calculate time array for PIV displacements.
    
    PIV correlates consecutive frame pairs to get N-1 displacements from N frames.
    Each displacement represents motion between frames i and i+1, timestamped at frame i+1.
    
    Args:
        frames (list[int]): List of frame numbers (e.g., [400, 401, ..., 799])
        dt (float): Time step between frames in seconds
        
    Returns:
        np.ndarray: Time array with N-1 elements for N frames
    """
    n_disps = len(frames) - 1
    return np.linspace(frames[0] * dt, (frames[0] + n_disps - 1) * dt, n_disps)


def backup(mode: str, proc_path: str, filename: str, var_names=None, test_mode=False, **kwargs) -> tuple[bool, dict]:
    """
    Load or save a backup file from/to the specified path.

    Args:
        mode (str): 'load' or 'save' to specify the operation.
        proc_path (str): Path to the directory containing the backup file.
        filename (str): Name of the backup file to load/save.
        test_mode (bool): If True, do not load/save the file.
        var_names (list): List of variable names to load (for load mode).
        **kwargs: Variables to save (for save mode). Use as: backup("save", path, file, var1=value1, var2=value2, ...)

    Returns:
        For load mode: loaded_vars (dict)
        For save mode: success (bool)
    """
    # If in test mode, return appropriate values
    if test_mode:
        return False, {}

    # Load mode
    elif mode == 'load':
        # Check if the file exists
        filepath = os.path.join(proc_path, filename)
        if not os.path.exists(filepath):
            print(f"Warning: backup file {filename} not found in {proc_path}.")
            return False, {}

        # Load the data from the .npz file
        else:
            loaded_vars = {}
            with np.load(filepath) as data:
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
                            print(f"Warning: {k} not found in {filepath}")
            print(f"Loaded data from {filepath}")
            return True, loaded_vars

    # Save mode
    elif mode == 'save':
        if not kwargs:
            print("Warning: No variables provided for saving.")
            return False, {}
        
        # Save the variables to a .npz file
        filepath = os.path.join(proc_path, filename)
        np.savez(filepath, **kwargs)
        print(f"Saved data to {filepath}")
        return True, {}

    # If mode is not recognized, return False
    else:
        print(f"Error: Unrecognized mode '{mode}'. Use 'load' or 'save'.")
        return False, {}


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


def read_imgs(data_path: str, frame_nrs: list[int] | str, format: str = 'tif', lead_0: int = 5, timing: bool = True) -> np.ndarray:

    """
    Load selected images from a directory into a 3D numpy array.

    Args:
        data_path (str): Path to the directory containing images.
        frame_nrs (list[int] | str): List of frame numbers to load,
            or "all" to load all images.
        format (str): File extension to load.
        lead_0 (int): Number of leading zeros in the file names.
        timing (bool): If True, show a progress bar while loading images.

    Returns:
        np.ndarray: 3D array of images (image_index, y, x).
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
        files = [f for f in files if f.endswith('.' + format) and not f.startswith('.')]
    else:
        # Filter files to include only those that match the specified frame numbers
        files = [f for f in files if any(f.endswith(f"{nr:0{lead_0}d}.{format}") for nr
                                         in frame_nrs) and not f.startswith('.')]
    
    if not files:
        raise FileNotFoundError(f"No files found in {data_path} with the specified criteria and format '{format}'.")

    # Read images into a 3D numpy array in parallel
    file_paths = [os.path.join(data_path, f) for f in files]
    
    n_jobs = os.cpu_count() or 4
    
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        imgs = list(tqdm(executor.map(read_img, file_paths), 
                        total=len(file_paths), 
                        desc='Reading images'))

    # Convert list of images to a numpy array
    imgs = np.array(imgs, dtype=np.uint64)
    return imgs


def downsample(imgs: np.ndarray, factor: int) -> np.ndarray:
    """Downsample a 2D image by summing non-overlapping blocks
     of size (block_size, block_size).

     Args:
        imgs (np.ndarray): 3D array of images (image_index, y, x).
        factor (int): Size of the blocks to sum over.

    Returns:
            np.ndarray: 3D array of downsampled images (image_index, y, x).
         """

    # Get image stack dimensions, check divisibility
    n_img, h, w = imgs.shape
    assert h % factor == 0 and w % factor == 0, \
        "Image dimensions must be divisible by block_size"

    # Reshape the image into blocks and sum over the blocks
    return imgs.reshape(n_img, h // factor, factor,
                        w // factor, factor).sum(axis=(2, 4))


def split_n_shift(img: np.ndarray, n_wins: tuple[int, int], overlap: float = 0, shift: tuple[int, int] | np.ndarray = (0, 0), shift_mode: str = 'before', plot: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a 2D image array (y, x) into (overlapping) windows,
    with automatic window size adjustments for shifted images.

    Args:
        img (np.ndarray): 2D array of image values (y, x).
        n_wins (tuple[int, int]): Number of windows in (y, x) direction.
        overlap (float): Fractional overlap between windows (0 = no overlap).
        shift (tuple[int, int] | np.ndarray): (dy, dx) shift in pixels - can be (0, 0) for uniform shift
                                              or 3D array (n_y, n_x, 2) for non-uniform shift per window.
        shift_mode (str): 'before' or 'after' shift: which frame is considered?
        plot (bool): If True, plot the windows on the image.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - wins: 4D array of image windows (window_y_idx, window_x_idx, y, x)
            - win_pos: corresponding positions (window_y_idx, window_x_idx, 2)
    """
    # Get dimensions
    img_h, img_w = img.shape
    n_y, n_x = n_wins

    # Handle both uniform and non-uniform shifts
    shift_array = np.asarray(shift, dtype=int)
    if shift_array.ndim == 1:  # Uniform shift (dy, dx)
        # Convert to non-uniform format
        dy, dx = shift_array
        shift_array = np.full((n_y, n_x, 2), [dy, dx])
    elif shift_array.shape != (n_y, n_x, 2):
        raise ValueError(
            f"Shift array must have shape ({n_y}, {n_x}, 2) for non-uniform shifts")

    # Calculate area from which to extract windows
    split_img_h = min(int(img_h // n_y * (1 + overlap)), img_h)
    split_img_w = min(int(img_w // n_x * (1 + overlap)), img_w)

    # Get the top-left corner of each window to create grid of window positions
    pos_y_idxs = np.linspace(0, img_h - split_img_h, num=n_y, dtype=int)
    pos_x_idxs = np.linspace(0, img_w - split_img_w, num=n_x, dtype=int)
    pos_grid = np.stack(np.meshgrid(
        pos_y_idxs, pos_x_idxs, indexing="ij"), axis=-1)

    # Compute physical centres of windows in image coordinates (for plotting/visualization)
    win_pos = np.stack((pos_grid[:, :, 0] + split_img_h / 2,
                        pos_grid[:, :, 1] + split_img_w / 2), axis=-1)

    # Determine cut-off direction: +1 for 'before', -1 for 'after'
    cut_off_dir = 1 if shift_mode == 'after' else -1

    # Show windows and centres on the image if requested
    if plot:
        fig, ax = plt.subplots()
        ax.imshow(img.astype(float) / img.max() * 255, cmap='gray')

    # Calculate window size after accounting for shifts
    win_h = split_img_h - np.max(np.abs(shift_array[:, :, 0]))
    win_w = split_img_w - np.max(np.abs(shift_array[:, :, 1]))

    # For each window...
    wins = np.zeros((n_y, n_x, win_h, win_w), dtype=img.dtype)
    for i, y in enumerate(pos_y_idxs):
        for j, x in enumerate(pos_x_idxs):

            # Get shift for this specific window
            dy, dx = shift_array[i, j]

            # Calculate cut-off for each direction for this window
            cut_y0 = max(0, cut_off_dir * dy)
            cut_y1 = max(0, -cut_off_dir * dy)
            cut_x0 = max(0, cut_off_dir * dx)
            cut_x1 = max(0, -cut_off_dir * dx)

            # Extract window with shift-specific cropping
            y0 = y + cut_y0
            y1 = y + split_img_h - cut_y1
            x0 = x + cut_x0
            x1 = x + split_img_w - cut_x1

            win_crop = img[y0:y1, x0:x1]

            # Crop to the smallest possible size
            win_h_crop, win_w_crop = win_crop.shape

            # If the current window is larger than target, crop it to target size
            if win_h_crop > win_h:
                excess_y = win_h_crop - win_h
                if cut_off_dir == 1 and dy < 0:  # 'after' mode with negative shift
                    win_crop = win_crop[excess_y:, :]
                else:
                    win_crop = win_crop[:-excess_y, :]

            if win_w_crop > win_w:
                excess_x = win_w_crop - win_w
                if cut_off_dir == 1 and dx < 0:  # 'after' mode with negative shift
                    win_crop = win_crop[:, excess_x:]
                else:
                    win_crop = win_crop[:, :-excess_x]

            # Now pad to reach exactly the target size
            win_h_crop, win_w_crop = win_crop.shape
            pad_y_needed = win_h - win_h_crop
            pad_x_needed = win_w - win_w_crop

            # Distribute padding to maintain feature alignment
            if cut_off_dir == 1:  # 'after' mode
                # Pad on the shift direction side
                pad_y_top = abs(dy) if dy > 0 and pad_y_needed > 0 else 0
                pad_x_left = abs(dx) if dx > 0 and pad_x_needed > 0 else 0
            else:  # 'before' mode
                # Pad on the opposite side to shift direction
                pad_y_top = abs(dy) if dy < 0 and pad_y_needed > 0 else 0
                pad_x_left = abs(dx) if dx < 0 and pad_x_needed > 0 else 0

            pad_y_bottom = max(0, pad_y_needed - pad_y_top)
            pad_x_right = max(0, pad_x_needed - pad_x_left)

            # Apply padding if needed
            if pad_y_needed > 0 or pad_x_needed > 0:
                wins[i, j] = np.pad(win_crop, ((pad_y_top, pad_y_bottom),
                                               (pad_x_left, pad_x_right)),
                                    mode='constant', constant_values=0)
            else:
                wins[i, j] = win_crop

            if plot:
                color = ['orange', 'blue'][(i + j) % 2]
                rect = plt.Rectangle((x + cut_x0, y + cut_y0),
                                     x1 - x0,
                                     y1 - y0,
                                     edgecolor=color, facecolor='none',
                                     linewidth=1.5)
                ax.add_patch(rect)
                ax.scatter(win_pos[i, j, 1], win_pos[i, j, 0], c=color,
                           marker='x', s=40)

    # Finish plot
    if plot:
        plt.xlim(-20, img_w + 20)
        plt.ylim(-20, img_h + 20)
        ax.set(
            title=f"{n_y}x{n_x} windows {shift_mode} shift ({100*overlap:.0f}% ov.)", xlabel='x', ylabel='y')

    return wins, win_pos


def disp2shift(n_wins: tuple[int, int], disp: np.ndarray) -> np.ndarray:
    """
    Distribute displacement values over a larger number of windows.
    
    This function takes displacement values from a smaller window grid and 
    distributes them to a larger window grid by replicating values across
    multiple windows.
    
    Args:
        n_wins (tuple[int, int]): Target number of windows (n_y, n_x)
        disp (np.ndarray): Displacement array with shape (n_frames, n_y_source, n_x_source, 2)
    
    Returns:
        np.ndarray: Shift array with shape (n_frames, n_y_target, n_x_target, 2)
                   compatible with split_n_shift function
    """
    n_y_target, n_x_target = n_wins
    
    # Validate input shape
    if disp.ndim != 4 or disp.shape[3] != 2:
        raise ValueError(f"Displacement array must have 4 dimensions (n_frames, n_y, n_x, 2), got {disp.shape}")
    
    n_frames, n_y_source, n_x_source, _ = disp.shape
    
    # Calculate how many target windows each source window should cover
    y_ratio = n_y_target / n_y_source
    x_ratio = n_x_target / n_x_source
    
    # Initialize output array
    shifts = np.zeros((n_frames, n_y_target, n_x_target, 2))
    
    # Distribute displacement values across target windows
    for i in range(n_y_source):
        for j in range(n_x_source):
            # Calculate target window range for this source window
            y0 = int(i * y_ratio)
            y1 = int((i + 1) * y_ratio)
            x0 = int(j * x_ratio)  
            x1 = int((j + 1) * x_ratio)
            
            # Assign the displacement to all target windows in this range
            shifts[:, y0:y1, x0:x1, :] = disp[:, i:i+1, j:j+1, :]
    
    return shifts


def calc_corr(i: int, imgs: np.ndarray, n_wins: tuple[int, int], shifts: np.ndarray, overlap: float) -> dict:
    """
    Calculate correlation maps for a single set of frames.

    Args:
        i (int): Frame index
        imgs (np.ndarray): 3D array of images (frame, y, x)
        n_wins (tuple[int, int]): Number of windows (n_y, n_x)
        shifts (np.ndarray): Array of shifts per frame (frame, y_shift, x_shift)
        overlap (float): Fractional overlap between windows (0 = no overlap)
        
    Returns:
        dict: Correlation maps for this set of frames as {(frame, win_y, win_x): (correlation_map, map_center)}
    """
    
    # Split images into windows with shifts
    wnd0, _ = split_n_shift(imgs[i], n_wins, shift=shifts[i],
                                  shift_mode='before', overlap=overlap)
    wnd1, _ = split_n_shift(imgs[i + 1], n_wins, shift=shifts[i],
                            shift_mode='after', overlap=overlap)

    # Calculate correlation maps and their centres for all windows
    corrs = {}
    for j in range(n_wins[0]):
        for k in range(n_wins[1]):

            # Correlate two (shifted) frames
            corr = sig.correlate(wnd1[j, k], wnd0[j, k],
                                     method='fft', mode='same')
            
            # Calculate the centre of the correlation map
            cntr = np.array(corr.shape) // 2

            # Store them in a dict
            corrs[(i, j, k)] = (corr, cntr)

    return corrs


def calc_corrs(imgs: np.ndarray, n_wins: tuple[int, int] = (1, 1), shifts: np.ndarray | None = None, overlap: float = 0, ds_fac: int = 1):
    """
    Calculate correlation maps for all frames and windows.

    Args:
        imgs (np.ndarray): 3D array of images (frame, y, x)
        n_wins (tuple[int, int]): Nr of windows (n_y, n_x)
        shifts (np.ndarray | None): 2D array of shifts per window
            (frame, y_shift, x_shift). If None, shift zero is used.
        overlap (float): Fractional overlap between windows (0 = no overlap)
        ds_fac (int): Downsampling factor (1 = no downsampling)

    Returns:
        dict: Correlation maps as {(frame, win_y, win_x):
            (correlation_map, map_center)}
    """
    n_corrs = len(imgs) - 1

    # Apply downsampling if needed
    if ds_fac > 1:
        imgs = downsample(imgs, ds_fac)

    # Handle shifts - default to zero if not provided
    if shifts is None:
        shifts = np.zeros((n_corrs, 2))

    # Prepare arguments for multithreading
    calc_corr_partial = partial(calc_corr, imgs=imgs, n_wins=n_wins, shifts=shifts, overlap=overlap)

    # Execute calc_corr in parallel for each frame
    n_jobs = os.cpu_count() or 4
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        frame_results = list(tqdm(
            executor.map(calc_corr_partial, range(n_corrs)), total=n_corrs, desc='Calculating correlation maps'))

    # Combine results from all frames
    corrs = {}
    for frame_result in frame_results:
        corrs.update(frame_result)

    return corrs


def sum_corr(i: int, corrs: dict, shifts: np.ndarray, n_tosum: int, n_wins: tuple[int, int], n_corrs: int) -> dict:
    """
    Sum correlation maps for a single set of frames.

    Args:
        i (int): Frame index
        corrs (dict): Correlation maps from calc_corrs
        shifts (np.ndarray): Array of shifts - can be 2D (frame, 2) or 4D (frame, win_y, win_x, 2)
        n_tosum (int): Number of correlation maps to sum
        n_wins (tuple[int, int]): Number of windows (n_y, n_x)
        n_corrs (int): Total number of correlation frames
        
    Returns:
        dict: Summed correlation maps for this frame as {(frame, win_y, win_x): (summed_map, new_center)}
    """
    
    # Calculate window bounds for summing: odd = symmetric, even = asymmetric
    i0 = max(0, i - (n_tosum - 1) // 2)
    i1 = min(n_corrs, i + n_tosum // 2 + 1)
    corr_idxs = np.arange(i0, i1)
    n_tosum = len(corr_idxs)
    
    # For each window...
    corrs_summed = {}
    for j in range(n_wins[0]):
        for k in range(n_wins[1]):
            
            # Single frame case - no alignment needed
            if n_tosum == 1:
                corr, _ = corrs[(corr_idxs[0], j, k)]
                corr_summed = corr
                cntr = np.array(corr_summed.shape) // 2
            
            # Multiple frames case - need alignment and summing
            else:
                # Extract shifts for this specific window
                if shifts.ndim == 2:  # 2D shifts: same for all windows
                    ref_shift = shifts[i]
                    window_shifts = shifts[corr_idxs]
                else:  # 4D shifts: different per window
                    ref_shift = shifts[i, j, k]
                    window_shifts = shifts[corr_idxs, j, k]
                
                # Calculate relative shifts for this window
                rel_shifts = (window_shifts - ref_shift).astype(int)
                
                # Collect all correlation maps for this window to determine shapes
                all_corrs = []
                all_shapes = []
                for frame_idx in corr_idxs:
                    corr, _ = corrs[(frame_idx, j, k)]
                    all_corrs.append(corr)
                    all_shapes.append(corr.shape)
                
                # Find the maximum shape needed to accommodate all correlation maps
                max_shape = (max(shape[0] for shape in all_shapes),
                            max(shape[1] for shape in all_shapes))
                
                # Calculate the expanded size needed to fit all shifted maps
                shift_min = np.min(rel_shifts, axis=0)
                shift_max = np.max(rel_shifts, axis=0)
                
                # Ensure we get scalar integers for shape calculations
                shift_range_y = int(shift_max[0] - shift_min[0])
                shift_range_x = int(shift_max[1] - shift_min[1])
                expanded_shape = (max_shape[0] + shift_range_y,
                                max_shape[1] + shift_range_x)
                
                # Pre-allocate the summed correlation map
                corr_summed = np.zeros(expanded_shape, dtype=all_corrs[0].dtype)
                
                # Calculate new center position in expanded map (based on max shape)
                cntr = (max_shape[0] // 2 - int(shift_min[0]),
                        max_shape[1] // 2 - int(shift_min[1]))
                
                # Sum each correlation map at its shifted position
                for corr, shift in zip(all_corrs, rel_shifts):
                    # Calculate placement indices
                    sy, sx = shift - shift_min
                    ey, ex = sy + corr.shape[0], sx + corr.shape[1]
                    
                    # Add correlation map to summed array
                    corr_summed[sy:ey, sx:ex] += corr

            # Store the summed map and its center for this window
            corrs_summed[(i, j, k)] = (corr_summed, cntr)
    
    return corrs_summed


def sum_corrs(corrs: dict, n_tosum: int, n_wins: tuple[int, int] = (1, 1), shifts: np.ndarray | None = None) -> dict:
    """
    Sum correlation maps with windowing and alignment.

    Args:
        corrs (dict): Correlation maps from calc_corrs
            as {(frame, win_y, win_x): (correlation_map, map_center)}
        n_tosum (int): Nr of corr. maps to sum (1 = none, even = asymmetric)
        n_wins (tuple[int, int]): Number of windows (n_y, n_x)
        shifts (np.ndarray | None): 2D array of shifts per window
            (frame, y_shift, x_shift). (0, 0, 0) if None

    Returns:
        dict: Summed correlation maps as {(frame, win_y, win_x, k): (summed_map, new_center)}
    """
    
    # Verify that n_tosum is a positive integer
    if n_tosum < 1 or not isinstance(n_tosum, int):
        raise ValueError("n_tosum must be a positive integer")

    # Determine number of frames from dictionary keys
    n_corrs = max(key[0] for key in corrs.keys()) + 1

    # Handle shifts - default to zero if not provided
    if shifts is None:
        shifts = np.zeros((n_corrs, 2))
    
    # Prepare arguments for multithreading
    sum_corr_partial = partial(sum_corr, corrs=corrs, shifts=shifts, n_tosum=n_tosum, n_wins=n_wins, n_corrs=n_corrs)
    
    n_jobs = os.cpu_count() or 4
    
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        frame_results = list(tqdm(executor.map(sum_corr_partial, range(n_corrs)), 
                                 total=n_corrs, 
                                 desc='Summing correlation maps'))
    
    # Combine results from all frames
    corrs_sum = {}
    for frame_result in frame_results:
        corrs_sum.update(frame_result)
    
    return corrs_sum
    

def find_peaks(corr: np.ndarray, n_peaks: int = 1, min_dist: int = 5, floor: float | None = None):

    """
    Find peaks in a correlation map.

    Args:
        corr (np.ndarray): 2D array of correlation values.
        n_peaks (int): Number of peaks to find.
        min_dist (int): Minimum distance between peaks in pixels.
        floor (float | None): Optional floor threshold for peak detection.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - peaks: Array of peak coordinates shaped (n_peaks, 2)
            - intensities: Intensities of the found peaks.
    """

    # Based on the median of the correlation map, set a floor
    if floor is not None:
        floor = floor * np.nanmedian(corr, axis=None)

        # # Check whether the floor is below the standard deviation
        # if floor < np.nanstd(corr_map, axis=None):
        #     print(f"Warning: floor {floor} is above the standard deviation.")

    if n_peaks == 1:
        # Find the single peak
        peaks = np.argwhere(np.amax(corr) == corr).astype(np.float64)
    else:
        # Find multiple peaks using peak_local_max
        peaks = peak_local_max(corr, min_distance=min_dist,
                               num_peaks=n_peaks, exclude_border=True, threshold_abs=floor).astype(np.float64)

    # If a smaller number of peaks is found, pad with NaNs
    if peaks.shape[0] < n_peaks:
        peaks = np.pad(peaks, ((0, n_peaks - peaks.shape[0]), (0, 0)),
                       mode='constant', constant_values=np.nan)

    # Calculate the intensities of the peaks
    ints = np.full(n_peaks, np.nan)
    
    # Only calculate intensities for valid (non-NaN) peaks
    valid_mask = ~np.isnan(peaks).any(axis=1)
    if np.any(valid_mask):
        valid_peaks = peaks[valid_mask]
        ints[valid_mask] = corr[valid_peaks[:, 0].astype(int), valid_peaks[:, 1].astype(int)]

    return peaks, ints


def three_point_gauss(array: np.ndarray) -> float:

    """
    Fit a Gaussian to three points.

    Args:
        array (np.ndarray): 1D array of three points, peak in the middle.

    Returns:
        float: Subpixel correction value.
    """

    # Check if the input is a 1D array
    if array.ndim != 1 or array.shape[0] != 3:
        raise ValueError("Input must be a 1D array with exactly three elements.")

    # Check if middle value is not the peak
    if array[1] < array[0] or array[1] < array[2]:
        raise ValueError("Middle value must be the peak of the three-point array.")
    
    # Shortcut for the symmetric case
    if array[0] == array[2]:
        return 0.0
    
    # Replace any zero values with 1 to avoid log(0) issues
    array1 = np.where(array <= 0, 1, array)
    
    # Calculate the denominator (PIV book §5.4.5)
    den = (np.log(array1[0]) + np.log(array1[2]) - 2 * np.log(array1[1]))
    
    # If the denominator is too small, return 0 to avoid division by zero
    if np.abs(den) < 1e-10:
        return 0.0
    else:
        return (0.5 * (np.log(array1[0]) - np.log(array1[2])) / den)


def subpixel(corr: np.ndarray, peak: np.ndarray) -> np.ndarray:

    """
    Use a Gaussian fit to refine the peak coordinates.

    Args:
        corr (np.ndarray): 2D array of correlation values.
        peak (np.ndarray): Coordinates of the peak (y, x).

    Returns:
        np.ndarray: Refined peak coordinates with subpixel correction.
    """

    # Apply three-point Gaussian fit to peak coordinates in two directions
    y_corr = three_point_gauss(corr[peak[0] - 1:peak[0] + 2, peak[1]])
    x_corr = three_point_gauss(corr[peak[0], peak[1] - 1:peak[1] + 2])

    # Add subpixel correction to the peak coordinates
    return peak.astype(np.float64) + np.array([y_corr, x_corr])


def find_disp(i: int, corrs: dict, shifts: np.ndarray, n_wins: tuple[int, int], n_peaks: int, ds_fac: int, subpx: bool = False, **find_peaks_kwargs) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Find peaks and calculate displacements for a single correlation map.
    
    Args:
        i (int): Correlation map index
        corrs (dict): Correlation maps as {(frame, win_y, win_x): (correlation_map, map_center)}
        shifts (np.ndarray): Array of shifts - can be 2D (frame, 2) or 4D (frame, win_y, win_x, 2)
        n_wins (tuple[int, int]): Number of windows (n_y, n_x)
        n_peaks (int): Number of peaks to find
        ds_fac (int): Downsampling factor to account for in displacement calculation
        subpx (bool): If True, apply subpixel accuracy using Gaussian fitting
        **find_peaks_kwargs: Additional arguments for find_peaks function (min_distance, floor, etc.)
        
    Returns:
        tuple: (frame_index, frame_disps, frame_ints) for this frame
    """
    
    # Initialize output arrays for this frame
    frame_disps = np.full((n_wins[0], n_wins[1], n_peaks, 2), np.nan)
    frame_ints = np.full((n_wins[0], n_wins[1], n_peaks), np.nan)
    
    for j in range(n_wins[0]):
        for k in range(n_wins[1]):
            # Get reference shift for this specific window
            if shifts.ndim == 2:  # 2D shifts: same for all windows
                ref_shift = shifts[i]
            else:  # 4D shifts: different per window
                ref_shift = shifts[i, j, k]
            
            # Get correlation map and its center (zero-displacement reference point)
            # NOTE: map_center is NOT the physical window position - it's the reference 
            # point in the correlation map coordinate system for calculating displacements
            corr_map, map_center = corrs[(i, j, k)]
            
            # Find peaks in the correlation map
            peaks, peak_ints = find_peaks(corr_map, n_peaks=n_peaks, 
                                               **find_peaks_kwargs)
            
            # Apply subpixel correction if requested
            if subpx:
                for p in range(n_peaks):
                    if not np.isnan(peaks[p]).any():  # Only apply to valid peaks
                        # Check if peak is not on the edge (needed for subpixel correction)
                        peak_y, peak_x = peaks[p].astype(int)
                        if (peak_y > 0 and peak_y < corr_map.shape[0] - 1 and 
                            peak_x > 0 and peak_x < corr_map.shape[1] - 1):
                            peaks[p] = subpixel(corr_map, peaks[p].astype(int))
            
            # Store intensities
            frame_ints[j, k, :] = peak_ints
            
            # Calculate displacements for all peaks
            frame_disps[j, k, :, :] = (ref_shift + 
                                          (peaks - map_center) * ds_fac)
    
    return i, frame_disps, frame_ints


def find_disps(corrs: dict, n_wins: tuple[int, int] = (1, 1), shifts: np.ndarray | None = None, n_peaks: int = 1, ds_fac: int = 1, subpx: bool = False, **find_peaks_kwargs) -> tuple[np.ndarray, np.ndarray]:
    """
    Find peaks in correlation maps and calculate displacements.

    Args:
        corrs (dict): Correlation maps 
            as {(frame, win_y, win_x): (correlation_map, map_center)}
        n_wins (tuple[int, int]): Number of windows (n_y, n_x)
        shifts (np.ndarray | None): 2D array of shifts per window (frame, y_shift, x_shift). If None, shift zero is used.
        n_peaks (int): Number of peaks to find
        ds_fac (int): Downsampling factor for  displacement calculation
        subpx (bool): If True, apply subpixel accuracy using Gaussian fitting
        **find_peaks_kwargs: Additional arguments for find_peaks function (min_distance, floor, etc.)

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - disps: 4D array (frame, win_y, win_x, peak, 2)
            - ints: 3D array (frame, win_y, win_x, peak)
    """
    
    # Determine number of frames from dictionary keys
    n_corrs = max(key[0] for key in corrs.keys()) + 1
    
    # Handle shifts - default to zero if not provided
    if shifts is None:
        shifts = np.zeros((n_corrs, 2))

    # Initialize output arrays
    disps = np.full((n_corrs, n_wins[0], n_wins[1], n_peaks, 2), np.nan)
    ints = np.full((n_corrs, n_wins[0], n_wins[1], n_peaks), np.nan)
    
    # Prepare arguments for multithreading
    find_disp_partial = partial(find_disp, corrs=corrs, shifts=shifts, n_wins=n_wins, n_peaks=n_peaks, ds_fac=ds_fac, subpx=subpx, **find_peaks_kwargs)

    n_jobs = os.cpu_count() or 4
    
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        frame_results = list(tqdm(executor.map(find_disp_partial, range(n_corrs)), 
                                 total=n_corrs, 
                                 desc='Finding peaks'))
    
    # Combine results from all frames
    for frame_idx, frame_disps, frame_ints in frame_results:
        disps[frame_idx] = frame_disps
        ints[frame_idx] = frame_ints
    
    return disps, ints
    

def filter_outliers(mode: str, coords: np.ndarray, a: float | np.ndarray | None = None, b: float | None = None, verbose: bool = False):

    """
    Remove outliers from coordinates based on spatial and intensity criteria.

    Args:
        mode (str): Filtering mode:
            - 'semicircle_rect': Filter a semicircle and rectangle
            - 'circle': Filter a circle at the origin
            - 'intensity': Filter based on intensity values of the provided peaks
        coords (np.ndarray): ND coordinate array of shape (..., 2)
        a (float | np.ndarray | None): Radius for filtering (or intensity values in 'intensity' mode)
        b (float | None): Width for rectangle filtering or relative threshold in 'intensity' mode  
        verbose (bool): If True, print summary of filtering.     

    Returns:
        np.ndarray | tuple[np.ndarray, np.ndarray]:
            - Filtered coordinates with invalid points set to NaN
            - (if mode == 'intensity') also returns filtered intensities
    """
    
    # Store original shape and flatten for processing
    orig_shape = coords.shape
    coords = coords.reshape(-1, coords.shape[-1])
    
    # OPTION 1: filter a semicircle of radius a (x < 0)
    # and a rectangle of height 2a and width b (x >= 0).
    if mode == 'semicircle_rect':
        if a is None or b is None:
            raise ValueError("Both 'a' and 'b' parameters must be provided for "
                             "'semicircle_rect' mode.")
        mask = (((coords[..., 1] < 0) &
                 (coords[..., 1] ** 2 + coords[:, 0] ** 2 <= a ** 2))
                | ((coords[:, 1] >= 0) & (coords[:, 1] <= b)
                   & (np.abs(coords[:, 0]) <= a)))
    
    # OPTION 2: filter a circle at the origin with radius a
    elif mode == 'circle':
        # TODO: Test circular filtering
        mask = (coords[:, 0] ** 2 + coords[:, 1] ** 2 <= a ** 2)

        if b is not None:
            print("Warning: 'b' parameter is ignored in 'circle' mode.")

    # OPTION 3: filter based on intensity values of the provided peaks
    elif mode == 'intensity':
        if a is None or b is None:
            raise ValueError("Both 'a' and 'b' parameters must be provided for "
                             "'intensity' mode.")
        
        # Check that a is an ND numpy array of intensity values
        if not isinstance(a, np.ndarray) or a.shape != orig_shape[:-1]:
            raise ValueError("Parameter 'a' must be an ND numpy array of intensity values matching the shape of the coordinates.")
        if not isinstance(b, (int, float)):
            raise ValueError("Parameter 'b' must be an integer or float representing the relative threshold.")

        # Reshape a to match the flattened coords
        ints = a.reshape(-1)

        # Calculate the intensity threshold
        int_min = b * np.nanmax(ints)

        # Create a mask based on the intensity threshold
        mask = (ints >= int_min)

    else:
        raise ValueError(f"Unknown filtering mode: {mode}")

    # Apply the mask to the coordinates*
    coords[~mask] = np.array([np.nan, np.nan])   
    
    if verbose:
        # Print summary statistics
        print(f"Post-processing: global filter removed {np.sum(~mask)}/{coords.shape[0]} coordinates in mode '{mode}'")

    # Reshape back to original shape
    coords = coords.reshape(orig_shape)
    
    # In intensity mode, also return the filtered intensities
    if mode == 'intensity':
        ints[~mask] = np.nan
        return coords, ints
    else:
        return coords


def cart2polar(coords: np.ndarray) -> np.ndarray:

    """
    Convert Cartesian coordinates to polar coordinates.

    Args:
        coords (np.ndarray): ND array of shape (..., 2) with (y, x) coordinates

    Returns:
        np.ndarray: ND array of shape (..., 2) with (r, phi) coordinates
    """
    # Calculate the magnitude and angle
    r = np.sqrt(coords[..., 0] ** 2 + coords[..., 1] ** 2)
    phi = np.arctan2(coords[..., 1], coords[..., 0])

    # Stack the results to form a new array
    polar_coords = np.stack((r, phi), axis=-1)
    return polar_coords


def validate_n_nbs(n_nbs: int | str | tuple[int, int, int], max_shape: tuple[int, int, int] | None = None):

    """
    Validate and process n_nbs parameter for filter_neighbours function.

    Args:
        n_nbs (int | str | tuple): Neighbourhood size specification (including center point)
            - int: Neighbourhood size in each dimension (must be odd).
            - str: "all" to use the full dimension length.
            - tuple: Three values specifying neighbourhood size in each dimension.
        max_shape (tuple[int, int, int] | None): Shape of the dimensions to use if n_nbs is "all"

    Returns:
        tuple[int, int, int]: Processed n_nbs values (neighbourhood sizes)
    """

    # Convert to list
    if isinstance(n_nbs, (int, str)):
        n_nbs = [n_nbs, n_nbs, n_nbs]
    elif isinstance(n_nbs, tuple):
        n_nbs = list(n_nbs)
    else:
        raise ValueError("n_nbs must be integer, 'all', or a tuple of three values (int or 'all').")
    
    # Process each dimension
    for i, n in enumerate(n_nbs):
        if n == "all":
            # Use dimension length (make it odd if necessary)
            n_nbs[i] = max_shape[i] - 1 if max_shape[i] % 2 == 0 else max_shape[i]
        elif isinstance(n, int):
            if n % 2 == 0:
                raise ValueError(f"n_nbs must be odd in each dimension (neighbourhood size including center). Got {n} for dimension {i}.")
        else:
            raise ValueError(f"Each element of n_nbs must be an integer or 'all'. Got {n} for dimension {i}.")
    
    return tuple(n_nbs)


def filter_neighbours(coords: np.ndarray, thr: float = 1, n_nbs: int | str | tuple[int, int, int] = 3, mode: str = "xy", replace: bool = False, verbose: bool = False):

    """
    Filter out coordinates that are too different from their neighbours.

    Args:
        coords (np.ndarray): 4D coordinate array of shape (n_corrs, n_wins_y, n_wins_x, 2).
        thr (float): Threshold; how many standard deviations can a point be away from its neighbours.
        n_nbs (int | str | tuple): Size of neighbourhood in each dimension to consider for filtering (including center point). Can be an integer, "all", or a tuple of three values (int or "all").
        mode (str): Which coordinates should be within std*thr from the median:
            - "x": Compare x coordinates only
            - "y": Compare y coordinates only
            - "xy": Compare both x and y coordinates
            - "r": Compare vector lengths only
        replace (bool): Replace outliers and pre-existing NaN values with the median of neighbours.
        verbose (bool): If True, print summary statistics about filtering.

    Returns:
        np.ndarray: Filtered coordinates with invalid points set to NaN or replaced with median.
    """

    # Get dimensions and validate n_nbs
    n_corrs, n_wins_y, n_wins_x, _ = coords.shape
    n_nbs = validate_n_nbs(n_nbs, (n_corrs, n_wins_y, n_wins_x))

    # Create a copy for output
    coords_output = coords.copy()
    
    # Initialize counters for verbose mode
    if verbose:
        outlier_count = 0
        nan_replaced_count = 0
        outlier_replaced_count = 0
    
    # Get a set of sliding windows around each coordinate
    # Note this function is slow
    nbs = np.lib.stride_tricks.sliding_window_view(coords,
    (n_nbs[0], n_nbs[1], n_nbs[2], 1))[..., 0]

    # Iterate over each coordinate
    for i in range(n_corrs):
        for j in range(n_wins_y):
            for k in range(n_wins_x):

                # First handle the coordinates at the edges, which are not in the centre of a neighbourhood
                i_nbs = (np.clip(i, (n_nbs[0] - 1)//2, 
                                 n_corrs - (n_nbs[0] - 1)//2 - 1) 
                         - (n_nbs[0] - 1)//2)
                j_nbs = (np.clip(j, (n_nbs[1] - 1)//2, 
                                 n_wins_y - (n_nbs[1] - 1)//2 - 1) 
                         - (n_nbs[1] - 1)//2)
                k_nbs = (np.clip(k, (n_nbs[2] - 1)//2, 
                                 n_wins_x - (n_nbs[2] - 1)//2 - 1) 
                         - (n_nbs[2] - 1)//2)
                nb = nbs[i_nbs, j_nbs, k_nbs]

                # If the neighbourhood is empty, skip to the next coordinate
                if np.all(np.isnan(nb)):
                    continue

                # If entire neighbourhood is identical, replace and skip
                if np.all(nb == nb[0, 0, 0, :]):
                    if replace:
                        coords_output[i, j, k, :] = nb[0, 0, 0, :]
                    continue

                # Calculate the median and standard deviation
                med = np.nanmedian(nb, axis=(1, 2, 3))
                std = np.nanstd(nb, axis=(1, 2, 3))
                
                # If std is 0 or NaN, skip outlier detection
                if np.any(np.isnan(std)) or np.any(std == 0):
                    continue

                # Check if the coordinate is already NaN in the input
                coord = coords[i, j, k, :]
                is_nan = np.any(np.isnan(coord))
                
                # Check if the current coordinate is an outlier
                is_outlier = False
                if not is_nan:
                    if mode == "x":
                        is_outlier = np.abs(coord[1] - med[1]) > thr * std[1]
                    elif mode == "y":
                        is_outlier = np.abs(coord[0] - med[0]) > thr * std[0]
                    elif mode == "xy":
                        is_outlier = np.any(np.abs(coord - med) > thr * std)
                    elif mode == "r":
                        vec_length = np.linalg.norm(coord)
                        med_length = np.linalg.norm(med)
                        is_outlier = np.abs(vec_length - med_length) > thr * std.mean()
                    else:
                        raise ValueError(f"Unknown mode: {mode}. Use 'x', 'y', 'xy', or 'r'.")
                
                # Update counters for verbose mode
                if verbose:
                    if is_outlier:
                        outlier_count += 1
                        if replace:
                            outlier_replaced_count += 1
                    if is_nan and replace:
                        nan_replaced_count += 1
                
                # Detailed verbose output (commented out for simplicity)
                # if verbose:
                #     status = "NaN" if is_nan else ("outlier" if is_outlier else "valid")
                #     print(f"Coordinate ({i}, {j}, {k}) is {status}: {coord}" +
                #           (f" (med: {med}, std: {std})" if status == "outlier" else ""))

                # Apply replacement or filtering logic
                if (replace and (is_nan or is_outlier)) or (not replace and is_outlier):
                    coords_output[i, j, k, :] = med if replace else np.array([np.nan, np.nan])

    # Print summary for verbose mode
    if verbose:
        if replace:
            print(f"Post-processing: neighbour filter replaced {outlier_replaced_count}/{len(coords_output)} outliers and {nan_replaced_count} other NaNs")
        else:
            print(f"Post-processing: neighbour filter removed {outlier_count}/{len(coords_output)} outliers")

    return coords_output


def first_valid(arr: np.ndarray) -> float | int | np.generic:

    """
    Function to find the first non-NaN value in a 1D array.

    Args:
        arr (np.ndarray): 1D array

    Returns:
        float | int | np.generic: First non-NaN value, or np.nan if none found
    """

    # Check if the input is a 1D array
    if arr.ndim == 1:
        for c in arr:
            if not np.isnan(c):
                return c
        # If no valid value found, return NaN
        return np.nan
    
    # Throw an error if the input is not 1D
    else:
        raise ValueError("Input must be a 1D array.")


def strip_peaks(coords: np.ndarray, axis: int = -2, verbose: bool = False) -> np.ndarray:

    """
    Reduce array dimensionality by selecting the first valid peak along an axis containing options.

    Args:
        coords (np.ndarray): N-D array where one axis represents different peaks
        axis (int): Axis along which to reduce the array (default: second-to-last axis)

    Returns:
        np.ndarray: (N-1)-D array with one axis reduced
    """

    if coords.ndim < 3:
        return coords  # Nothing to strip
    
    # Apply the first_valid function along the specified axis  
    coords_str = np.apply_along_axis(first_valid, axis, coords.copy())

    # Report on the number of NaNs
    if verbose:
        n_nans_i = np.sum(np.any(np.isnan(coords[:, :, :, 0, :]), axis=-1))
        n_nans_f = np.sum(np.any(np.isnan(coords_str), axis=-1))

        print(f"Post-processing: {n_nans_i}/{np.prod(coords.shape[0:3])} most likely peak candidates invalid; left with {n_nans_f} after taking next-best peak")
    return coords_str


def smooth(time: np.ndarray, disps: np.ndarray, col: str | int = 'both', lam: float = 5e-7, type: type = int) -> np.ndarray:

    """
    Smooth displacement data along a specified axis using a smoothing spline.

    Args:
        time (np.ndarray): 1D array of time values.
        disps (np.ndarray): 2D array of displacement values.
        col (str | int): Column to smooth:
            - 'both': Smooth both columns (y and x displacements).
            - int: Index of the column to smooth (0 for y, 1 for x).
        lam (float): Smoothing parameter. Larger = more smoothing.
        type (type): Type to convert the smoothed displacements to.

    Returns:
        np.ndarray: 2D array of smoothed displacements (same shape as input)
    """

    # Work on copy
    disps_spl = disps.copy()
    orig_shape = disps_spl.shape

    # Try to squeeze displacements array, then check if 2D
    disps_spl = disps_spl.squeeze() if disps_spl.ndim > 2 else disps_spl
    if disps_spl.ndim != 2:
        raise ValueError("disps must be a 2D array with shape (n_time, 2).")

    # Mask any NaN values in the displacements
    mask = ~np.isnan(disps_spl).any(axis=1)

    # If cols is 'both', apply smoothing to both columns
    if col == 'both':
        for i in range(disps_spl.shape[1]):
            disps_spl[:, i] = make_smoothing_spline(time[mask], disps_spl[mask, i], lam=lam)(time).astype(type)

    # Otherwise, apply smoothing to the specified column
    elif isinstance(col, int):
        disps_spl[:, col] = make_smoothing_spline(time[mask], disps_spl[mask, col], lam=lam)(time).astype(type)
    else:
        raise ValueError("cols must be 'both' or an integer index.")
    
    return disps_spl.reshape(orig_shape)


def plot_vel_comp(disp_glo, disp_nbs, disp_spl, res, frs, dt, proc_path=None, file_name=None, test_mode=False, **kwargs):
    # TODO Add docstring and typing
    # Might break with horizontal windows.

    # Define a time array using helper function
    time = get_time(frs, dt)

    # If lengths don't match, assume all data was supplied; slice accordingly
    if disp_glo.shape[0] != time.shape[0]:
        disp_glo = disp_glo[frs[0]:frs[-1], :, :, :]
        disp_nbs = disp_nbs[frs[0]:frs[-1], :, :, :]
        disp_spl = disp_spl[frs[0]:frs[-1], :, :, :]

    # Convert displacement to velocity
    vel_glo = disp_glo * res / dt
    vel_nbs = disp_nbs * res / dt
    vel_spl = disp_spl * res / dt

    # Scatter plot vx(t)
    fig, ax = plt.subplots(figsize=(10, 6))

    # ax.plot(np.tile(time[:, None] * 1000, (1, n_peaks)).flatten(),
    #         vel_unf[:, 0, 0, :, 1].flatten(), 'x', c='gray', alpha=0.5, ms=4, label='vx (all candidate peaks)')
    # ax.plot(1000 * time, vel_unf[:, 0, 0, 0, 1].flatten(), 'x', c='gray', alpha=0.5, ms=4, label='vx (brightest peak)')

    ax.plot(1000 * time, vel_glo[:, 0, 0, 1], 'o', ms=4, c='gray',
            label='vx (filtered globally)')
    ax.plot(1000 * time, vel_nbs[:, 0, 0, 1], '.', ms=2, c='black',
            label='vx (filtered neighbours)')
    ax.plot(1000 * time, vel_nbs[:, 0, 0, 0], c=cvd.get_color(0), 
            label='vy (filtered neighbours)')
    ax.plot(1000 * time, vel_spl[:, 0, 0, 1], c=cvd.get_color(1),
        label='vx (smoothed for 2nd pass)')

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('First pass')
    ax.set(**kwargs)

    ax.legend()
    ax.grid()

    if proc_path is not None and file_name is not None and not test_mode:
        # Save the figure
        save_cfig(proc_path, file_name, test_mode=test_mode, verbose=True)

    return fig, ax


def plot_vel_med(disp, res, frs, dt, proc_path=None, file_name=None, test_mode=False, **kwargs):
    # TODO Add docstring and typing
    # Might break with horizontal windows.

    # Define a time array
    time = get_time(frs, dt)

    # If lengths don't match, assume all data was supplied; slice accordingly
    if disp.shape[0] != time.shape[0]:
        disp = disp[frs[0]:frs[-1], :, :, :]

    # Convert displacement to velocity
    vel = disp * res / dt

    # Plot the median velocity in time, show the min and max as a shaded area
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot vy (vertical velocity)
    ax.plot(time * 1000,
            np.nanmedian(vel[:, :, :, 0], axis=(1, 2)), label='Median vy')
    ax.fill_between(time * 1000,
                    np.nanmin(vel[:, :, :, 0], axis=(1, 2)),
                    np.nanmax(vel[:, :, :, 0], axis=(1, 2)),
                    alpha=0.3, label='Min/max vy')

    # Plot vx (horizontal velocity)
    ax.plot(time * 1000,
            np.nanmedian(vel[:, :, :, 1], axis=(1, 2)), label='Median vx')
    ax.fill_between(time * 1000,
                    np.nanmin(vel[:, :, :, 1], axis=(1, 2)),
                    np.nanmax(vel[:, :, :, 1], axis=(1, 2)),
                    alpha=0.3, label='Min/max vx')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Median velocity in time')
    ax.set(**kwargs)

    ax.legend()
    ax.grid()

    if proc_path is not None and file_name is not None and not test_mode:
        # Save the figure
        save_cfig(proc_path, file_name, test_mode=test_mode, verbose=True)

    return fig, ax


def plot_vel_prof(disp, res, frs, dt, win_pos, 
                  mode="random", proc_path=None, file_name=None, subfolder=None, test_mode=False, **kwargs):
    # TODO: Write docstring
    
    # Define a time array
    n_corrs = disp.shape[0]
    time = get_time(frs, dt)
    
    # Convert displacement to velocity
    vel = disp * res / dt
  
    # Raise error if one tries to make a video, but proc_path is not specified
    if mode == "video" and (proc_path is None or file_name is None
                             or test_mode):
        raise ValueError("proc_path and file_name must be specified, and test_mode must be False to create a video.")

    # Set up save path if subfolder is specified
    if proc_path is not None and subfolder is not None and not test_mode:
        save_path = os.path.join(proc_path, subfolder)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        if mode == "all":
            # Error: we don't want to save all images to the root folder
            raise RuntimeWarning(f"Are you sure you want to save {n_corrs} files directly to {proc_path}?")
        save_path = proc_path
    
    # Determine which frames to process
    if mode == "random":
        np.random.seed(42)  # For reproducible results
        frames_to_plot = np.sort(np.random.choice(n_corrs, size=min(10, n_corrs), replace=False))
    elif mode == "all" or mode == "video":
        frames_to_plot = range(n_corrs)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'video', 'all', or 'random'.")
    
    # Set up video writer if needed
    if mode == "video":
        fig_video, ax_video = plt.subplots(figsize=(10, 6))
        writer = ani.FFMpegWriter(fps=10)
        video_path = os.path.join(proc_path, file_name+'.mp4')
        video_context = writer.saving(fig_video, video_path, dpi=150)
        frames_iter = trange(n_corrs, desc='Creating velocity profile video')
    else:
        video_context = None
        frames_iter = frames_to_plot
    
    # Common plotting function
    def plot_frame(frame_idx, ax):
        y_pos = win_pos[:, 0, 0] * res * 1000
        vx = vel[frame_idx, :, 0, 1]
        vy = vel[frame_idx, :, 0, 0]
        
        ax.plot(vx, y_pos, '-o', c=cvd.get_color(1), label='vx')
        ax.plot(vy, y_pos, '-o', c=cvd.get_color(0), label='vy')
        ax.set_xlabel('Velocity (m/s)')
        ax.set_ylabel('y position (mm)')
        ax.set_title(f'Velocity profiles at frame {frame_idx + 1} ({time[frame_idx] * 1000:.2f} ms)')
        ax.legend()
        ax.grid()
        ax.set(**kwargs)
    
    # Process frames
    if video_context is not None:
        # Video mode
        with video_context:
            for i in frames_iter:
                ax_video.clear()
                plot_frame(i, ax_video)
                writer.grab_frame()
        plt.close(fig_video)
        print(f"Video saved to {video_path}")
    else:
        # Plot mode (random or all)
        for frame_idx in frames_iter:
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_frame(frame_idx, ax)
            
            # Save if path is specified
            if save_path is not None:
                save_cfig(save_path, file_name + f"_{frame_idx:04d}", test_mode=test_mode)
                
                # Close figure
                plt.close(fig)


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