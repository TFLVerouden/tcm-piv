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
            corr = sig.correlate(wnd1[j, k].astype(np.uint32), wnd0[j, k].astype(np.uint32),
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
    calc_corr_partial = partial(
        calc_corr, imgs=imgs, n_wins=n_wins, shifts=shifts, overlap=overlap)

    # Execute calc_corr in parallel for each frame
    n_jobs = os.cpu_count() or 4
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        frame_results = list(tqdm(
            executor.map(calc_corr_partial, range(n_corrs)), total=n_corrs, desc='Correlating windows '))

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
                corr_summed = np.zeros(
                    expanded_shape, dtype=all_corrs[0].dtype)

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
    sum_corr_partial = partial(
        sum_corr, corrs=corrs, shifts=shifts, n_tosum=n_tosum, n_wins=n_wins, n_corrs=n_corrs)

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
