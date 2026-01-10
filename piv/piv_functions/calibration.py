"""
Calibration functions for PIV analysis.

This module handles camera calibration using grid patterns to determine
pixel-to-real-world resolution.
"""

import numpy as np
import cv2 as cv
from scipy import spatial
import os
from matplotlib import pyplot as plt
from .io import save_backup, save_cfig


def all_distances(points):
    """
    Calculate pairwise Euclidean distances between all points.
    
    Args:
        points (np.ndarray): Array of points with shape (n_points, 2).
        
    Returns:
        np.ndarray: Distance matrix with shape (n_points, n_points).
        
    Raises:
        ValueError: If input points don't have the correct shape.
    """
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("Input points must be a 2D array with shape (n_points, 2)")
    return spatial.distance.cdist(points, points, 'euclidean')


def calibrate_grid(path, spacing, roi=None, init_grid=(4, 4), binary_thr=100,
                   blur_ker=(3, 3), open_ker=(3, 3), print_prec=5,
                   plot=False, save=True):
    """
    Calculate resolution from a grid pattern image.

    This function detects a symmetric grid of circles in a calibration image
    and calculates the pixel-to-real-world resolution. It saves comprehensive
    calibration data including original file information, ROI data, processing
    parameters, and results to an npz file.

    Parameters:
        path (str): Path to the calibration image file.
        spacing (float): Real-world spacing between grid points [m].
        roi (list): Region to crop to (y_start, y_end, x_start, x_end).
                    If None, the entire image is used.
        init_grid (tuple): Initial grid size (columns, rows) to start detection.
        binary_thr (int): Threshold value for binarising the image.
        blur_ker (tuple): Kernel size for Gaussian blur (width, height).
        open_ker (tuple): Kernel size for morphological opening (width, height).
        print_prec (int): Number of decimal places for printing resolution.
        plot (bool): Whether to display a plot of the detected grid.
        save (bool): Whether to save calibration data to npz file.

    Returns:
        tuple: (res_avg, res_std) - Average and standard deviation of resolution [m/px].
    """

    # Load the image and convert it to grayscale
    img = cv.imread(path, cv.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")

    # If the number of channels is greater than 1, convert to grayscale
    if img.ndim > 2 and img.shape[2] > 1:
        img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    orig_shape = img.shape

    # Crop the image to the specified region of interest (ROI)
    if roi is None:
        roi = [0, img.shape[0], 0, img.shape[1]]
    img = img[roi[0]:roi[1], roi[2]:roi[3]]

    # Apply morphological opening and Gaussian blur to preprocess the image
    img = cv.morphologyEx(img, cv.MORPH_OPEN, np.ones(open_ker, np.uint8))
    img = cv.GaussianBlur(img, blur_ker, 0)

    # Binarize the image using the specified threshold
    img = np.where(img > binary_thr, 255, img)

    # Initialize grid size and flag for maximum columns found
    grid_size = list(init_grid)
    max_columns_found = False

    # Loop to find the largest fitting grid size
    while True:
        grid_found, centres = cv.findCirclesGrid(
            img, tuple(grid_size), flags=cv.CALIB_CB_SYMMETRIC_GRID)
        
        # Print the current grid size being tested
        print(f"Trying {grid_size[1]} × {grid_size[0]} grid...", end='\r')

        if grid_found and not max_columns_found:
            # Increase the number of columns if the maximum hasn't been reached
            grid_size[0] += 1
        elif not grid_found and not max_columns_found:
            # Fix the number of columns and start increasing rows
            max_columns_found = True
            grid_size[0] -= 1
            grid_size[1] += 1
        elif max_columns_found:
            # Only increase the number of rows
            grid_found, centres = cv.findCirclesGrid(
                img, tuple(grid_size), flags=cv.CALIB_CB_SYMMETRIC_GRID)
            if grid_found:
                grid_size[1] += 1
            else:
                # Revert to the last successful row count and exit the loop
                grid_size[1] -= 1
                break

    print(f"Grid found: {grid_size[1]} cols × {grid_size[0]} rows")

    # Reshape the detected centers and generate grid points in real-world units
    centres = centres.reshape(-1, 2)
    grid_points = np.array([[x, y] for y in range(grid_size[1])
                            for x in range(grid_size[0])]) * spacing

    # Calculate pairwise distances in real-world and pixel units
    dist_real = all_distances(grid_points)
    dist_pixel = all_distances(centres)

    # Compute resolution and standard deviation
    with np.errstate(divide='ignore', invalid='ignore'):
        all_res = dist_real / dist_pixel

    mask = np.eye(*dist_pixel.shape, dtype=bool).__invert__()
    res_avg = np.average(all_res[mask], weights=dist_pixel[mask])
    res_std = np.sqrt(np.average((all_res[mask]-res_avg)**2, weights=dist_pixel[mask]))

    # Print the resolution and standard deviation
    print(f"Resolution (± std): {res_avg*1000:.{print_prec}f} ± "
          f"{res_std*1000:.{print_prec}f} mm/px")

    # Calculate calibration image size
    frame_size = np.array([orig_shape[0] * res_avg, orig_shape[1] * res_avg])  # in m
    print(f"Frame size: {frame_size[0]*1000:.{print_prec//2}f} × "
          f"{frame_size[1]*1000:.{print_prec//2}f} mm²")

    # Save comprehensive calibration data to npz file
    if save:     
        # Save all calibration data
        save_backup(proc_path=path,
                    file_name=path.replace('.tif', '.npz'),
                    # Original file information
                    original_file_path=path,
                    original_image_shape=orig_shape,
                    # ROI information
                    roi_used=np.array(roi),
                    # Calibration parameters
                    spacing_m=spacing,
                    init_grid=np.array(init_grid),
                    binary_threshold=binary_thr,
                    blur_kernel=np.array(blur_ker),
                    open_kernel=np.array(open_ker),
                    # Resolution results
                    resolution_avg_m_per_px=res_avg,
                    resolution_std_m_per_px=res_std,
                    resolution_avg_mm_per_px=res_avg * 1000,
                    resolution_std_mm_per_px=res_std * 1000,
                    # Frame size in m
                    frame_size_m=frame_size,
                    # Grid detection results
                    final_grid_size=np.array(grid_size),
                    grid_centres_pixel=centres,
                    grid_points_real=grid_points,
                    # Distance matrices for reference
                    distance_real_m=dist_real,
                    distance_pixel_px=dist_pixel,
                    )    

    # Optionally plot the results
    if plot:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.imshow(img, cmap='gray')
        ax.scatter(centres[:, 0], centres[:, 1], c='g', s=30, alpha=0.7)
        ax.set_title(f"Grid Calibration: {grid_size[0]} cols × {grid_size[1]} rows\n"
                     f"Resolution: {res_avg*1000:.{print_prec}f} mm/px")
        ax.set_xlabel('Pixels')
        ax.set_ylabel('Pixels')
        plt.tight_layout()
        plt.show()
        
        if save:
            save_cfig(path, path.replace('.tif', ''))

    print("Calibration complete.")

    return res_avg, res_std, frame_size

if __name__ == "__main__":
    # CALIBRATION 250624
    # Get the directory containing the file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cal_path = os.path.join(current_dir, "calibration",
                            "250624_calibration_PIV_500micron.tif")

    # Define calibration parameters
    cal_spacing = 0.0005  # m
    cal_roi = [50, 725, 270, 375]

    # Run the calibration function
    calibrate_grid(cal_path, cal_spacing, roi=cal_roi, save=True, plot=True)

    # CALIBRATION 250723
    cal_path = os.path.join(current_dir, "calibration",
                            "calibration_PIV_500micron_2025_07_23_C001H001S0001.tif")
    
    # Adjust calibration parameters
    cal_roi = [45, 825, 225, 384]
    
    # Re-run the calibration function
    calibrate_grid(cal_path, cal_spacing, roi=[45, 825, 225, 384], init_grid=(7, 5), binary_thr=200, blur_ker=(5, 5), save=True, plot=True)