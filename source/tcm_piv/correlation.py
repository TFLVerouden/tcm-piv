"""
Cross-correlation functions for PIV analysis.

This module contains functions for calculating and summing
cross-correlation maps between image windows.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
from scipy import signal as sig
from tqdm import tqdm

# Import functions from other submodules
from .preprocessing import split_n_shift, downsample


def calc_corr(i: int, imgs: np.ndarray, n_windows: tuple[int, int], shifts: np.ndarray, overlap: float) -> dict:
    """
    Calculate correlation maps for a single set of frames.

    Args:
        i (int): Frame index
        imgs (np.ndarray): 3D array of images (frame, y, x)
        n_windows (tuple[int, int]): Number of windows (n_y, n_x)
        shifts (np.ndarray): Array of shifts per frame (frame, y_shift, x_shift)
        overlap (float): Fractional overlap between windows (0 = no overlap)

    Returns:
        dict: Correlation maps for this set of frames as {(frame, win_y, win_x): (correlation_map, map_center)}
    """

    # Split images into windows with shifts
    windows0, _ = split_n_shift(imgs[i], n_windows, shift=shifts[i],
                                shift_mode='before', overlap=overlap)
    windows1, _ = split_n_shift(imgs[i + 1], n_windows, shift=shifts[i],
                                shift_mode='after', overlap=overlap)

    # Calculate correlation maps and their centres for all windows
    corr_maps = {}
    for j in range(n_windows[0]):
        for k in range(n_windows[1]):

            # Correlate two (shifted) frames
            corr = sig.correlate(windows1[j, k].astype(np.uint32), windows0[j, k].astype(np.uint32),
                                 method='fft', mode='same')

            # Calculate the centre of the correlation map
            corr_center_yx = np.array(corr.shape) // 2

            # Store them in a dict
            corr_maps[(i, j, k)] = (corr, corr_center_yx)

    return corr_maps


def calc_corrs(imgs: np.ndarray, n_windows: tuple[int, int] = (1, 1), shifts: np.ndarray | None = None, overlap: float = 0, ds_factor: int = 1):
    """
    Calculate correlation maps for all frames and windows.

    Args:
        imgs (np.ndarray): 3D array of images (frame, y, x)
        n_windows (tuple[int, int]): Nr of windows (n_y, n_x)
        shifts (np.ndarray | None): 2D array of shifts per window
            (frame, y_shift, x_shift). If None, shift zero is used.
        overlap (float): Fractional overlap between windows (0 = no overlap)
        ds_factor (int): Downsampling factor (1 = no downsampling)

    Returns:
        dict: Correlation maps as {(frame, win_y, win_x):
            (correlation_map, map_center)}
    """
    n_corrs = len(imgs) - 1

    # Apply downsampling if needed
    if ds_factor > 1:
        imgs = downsample(imgs, ds_factor)

    # Handle shifts - default to zero if not provided
    if shifts is None:
        shifts = np.zeros((n_corrs, 2))

    # Prepare arguments for multithreading
    calc_corr_partial = partial(
        calc_corr, imgs=imgs, n_windows=n_windows, shifts=shifts, overlap=overlap)

    # Execute calc_corr in parallel for each frame
    n_jobs = os.cpu_count() or 4
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        frame_results = list(tqdm(
            executor.map(calc_corr_partial, range(n_corrs)), total=n_corrs, desc='Correlating windows '))

    # Combine results from all frames
    corr_maps = {}
    for frame_result in frame_results:
        corr_maps.update(frame_result)

    return corr_maps


def sum_corr(i: int, corrs: dict, shifts: np.ndarray, n_corrs_to_sum: int, n_windows: tuple[int, int], n_corrs: int) -> dict:
    """
    Sum correlation maps for a single set of frames.

    Args:
        i (int): Frame index
        corrs (dict): Correlation maps from calc_corrs
        shifts (np.ndarray): Array of shifts - can be 2D (frame, 2) or 4D (frame, win_y, win_x, 2)
        n_corrs_to_sum (int): Number of correlation maps to sum
        n_windows (tuple[int, int]): Number of windows (n_y, n_x)
        n_corrs (int): Total number of correlation frames

    Returns:
        dict: Summed correlation maps for this frame as {(frame, win_y, win_x): (summed_map, new_center)}
    """

    # Calculate window bounds for summing: odd = symmetric, even = asymmetric
    i0 = max(0, i - (n_corrs_to_sum - 1) // 2)
    i1 = min(n_corrs, i + n_corrs_to_sum // 2 + 1)
    corr_indices = np.arange(i0, i1)
    n_corrs_to_sum = len(corr_indices)

    # For each window...
    corr_maps_summed = {}
    for j in range(n_windows[0]):
        for k in range(n_windows[1]):

            # Single frame case - no alignment needed
            if n_corrs_to_sum == 1:
                corr, _ = corrs[(corr_indices[0], j, k)]
                corr_sum = corr
                corr_center_yx = np.array(corr_sum.shape) // 2

            # Multiple frames case - need alignment and summing
            else:
                # Extract shifts for this specific window
                if shifts.ndim == 2:  # 2D shifts: same for all windows
                    ref_shift = shifts[i]
                    window_shifts = shifts[corr_indices]
                else:  # 4D shifts: different per window
                    ref_shift = shifts[i, j, k]
                    window_shifts = shifts[corr_indices, j, k]

                # Calculate relative shifts for this window
                rel_shifts_yx = (window_shifts - ref_shift).astype(int)

                # Collect all correlation maps for this window to determine shapes
                all_corrs = []
                all_shapes = []
                for frame_idx in corr_indices:
                    corr, _ = corrs[(frame_idx, j, k)]
                    all_corrs.append(corr)
                    all_shapes.append(corr.shape)

                # Find the maximum shape needed to accommodate all correlation maps
                max_shape = (max(shape[0] for shape in all_shapes),
                             max(shape[1] for shape in all_shapes))

                # Calculate the expanded size needed to fit all shifted maps
                shift_min = np.min(rel_shifts_yx, axis=0)
                shift_max = np.max(rel_shifts_yx, axis=0)

                # Ensure we get scalar integers for shape calculations
                shift_range_y = int(shift_max[0] - shift_min[0])
                shift_range_x = int(shift_max[1] - shift_min[1])
                expanded_shape = (max_shape[0] + shift_range_y,
                                  max_shape[1] + shift_range_x)

                # Pre-allocate the summed correlation map
                corr_sum = np.zeros(
                    expanded_shape, dtype=all_corrs[0].dtype)

                # Calculate new center position in expanded map (based on max shape)
                corr_center_yx = (max_shape[0] // 2 - int(shift_min[0]),
                                  max_shape[1] // 2 - int(shift_min[1]))

                # Sum each correlation map at its shifted position
                for corr, shift in zip(all_corrs, rel_shifts_yx):
                    # Calculate placement indices
                    sy, sx = shift - shift_min
                    ey, ex = sy + corr.shape[0], sx + corr.shape[1]

                    # Add correlation map to summed array
                    corr_sum[sy:ey, sx:ex] += corr

            # Store the summed map and its center for this window
            corr_maps_summed[(i, j, k)] = (corr_sum, corr_center_yx)

    return corr_maps_summed


def sum_corrs(corrs: dict, n_corrs_to_sum: int, n_windows: tuple[int, int] = (1, 1), shifts: np.ndarray | None = None) -> dict:
    """
    Sum correlation maps with windowing and alignment.

    Args:
        corrs (dict): Correlation maps from calc_corrs
            as {(frame, win_y, win_x): (correlation_map, map_center)}
        n_corrs_to_sum (int): Nr of corr. maps to sum (1 = none, even = asymmetric)
        n_windows (tuple[int, int]): Number of windows (n_y, n_x)
        shifts (np.ndarray | None): 2D array of shifts per window
            (frame, y_shift, x_shift). (0, 0, 0) if None

    Returns:
        dict: Summed correlation maps as {(frame, win_y, win_x): (summed_map, new_center)}
    """

    # Verify that n_tosum is a positive integer
    if n_corrs_to_sum < 1 or not isinstance(n_corrs_to_sum, int):
        raise ValueError("n_corrs_to_sum must be a positive integer")

    # Determine number of frames from dictionary keys
    n_corrs = max(key[0] for key in corrs.keys()) + 1

    # Handle shifts - default to zero if not provided
    if shifts is None:
        shifts = np.zeros((n_corrs, 2))

    # Prepare arguments for multithreading
    sum_corr_partial = partial(
        sum_corr, corrs=corrs, shifts=shifts, n_corrs_to_sum=n_corrs_to_sum, n_windows=n_windows, n_corrs=n_corrs)

    n_jobs = os.cpu_count() or 4

    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        frame_results = list(tqdm(executor.map(sum_corr_partial, range(n_corrs)),
                                  total=n_corrs,
                                  desc='Summing correlations'))

    # Combine results from all frames
    corrs_sum = {}
    for frame_result in frame_results:
        corrs_sum.update(frame_result)

    return corrs_sum
