"""
Displacement calculation functions for PIV analysis.

This module contains functions for finding correlation peaks
and calculating displacements from correlation maps.
"""

import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
from skimage.feature import peak_local_max
from tqdm import tqdm


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


def find_disps(corrs: dict, n_wins: tuple[int, int] = (1, 1), shifts: np.ndarray | None = None, n_peaks: int = 1, ds_fac: int = 1, subpx: bool = False, verbose: bool = True, **find_peaks_kwargs) -> tuple[np.ndarray, np.ndarray]:
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
                                 desc='Finding peaks       '))
    
    # Combine results from all frames
    for frame_idx, frame_disps, frame_ints in frame_results:
        disps[frame_idx] = frame_disps
        ints[frame_idx] = frame_ints
    
    # If verbose, print how many displacements were not found
    if verbose:
        n_nan_disps = np.sum(np.all(np.isnan(disps), axis=(-2, -1)))
        print(f"Finding peaks: {n_nan_disps}/{n_corrs * n_wins[0] * n_wins[1]} windows did not yield any displacements.")
    return disps, ints
