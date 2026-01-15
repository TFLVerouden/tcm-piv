"""
Image preprocessing functions for PIV analysis.

This module contains functions for preparing images for PIV analysis,
including downsampling and splitting images into interrogation windows.
"""

import numpy as np
from matplotlib import pyplot as plt


def generate_background(imgs: np.ndarray, method: str = 'median') -> np.ndarray:
    """Generate a background image from a stack of images.

    Args:
        imgs (np.ndarray): 3D array of images (image_index, y, x).
        method (str): Method to compute background ('median' or 'mean').

    Returns:
        np.ndarray: 2D background image (y, x).
    """
    if method == 'median':
        background = np.median(imgs, axis=0)
    elif method == 'mean':
        background = np.mean(imgs, axis=0)
    else:
        raise ValueError("Method must be 'median' or 'mean'.")
    return background.astype(imgs.dtype)


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
    n_images, height, width = imgs.shape
    assert height % factor == 0 and width % factor == 0, \
        "Image dimensions must be divisible by block_size"

    # Reshape the image into blocks and sum over the blocks
    return imgs.reshape(n_images, height // factor, factor,
                        width // factor, factor).sum(axis=(2, 4))


def crop(imgs: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    """"
    Crop a single image or a stack of images.

    Args:
        imgs (np.ndarray): 2D image (y, x) or 3D stack (image_index, y, x).
        roi (tuple[int, int, int, int]): Region of interest as (y_start, y_end, x_start, x_end). If y_end or x_end is negative, they are interpreted as an offset from the end of the image. If y_end or x_end is zero, it is interpreted as the full extent in that direction.

    Returns:
        np.ndarray: Cropped image or stack of images.
    """

    # Check if the input is a 3D array
    if imgs.ndim != 3:
        if imgs.ndim == 2:
            # If it's a 2D image, add a new axis to make it 3D
            imgs = imgs[np.newaxis, ...]
            was_2d = True
        else:
            raise ValueError(
                "Input must be a 3D array of images (image_index, y, x).")

    # Validate ROI dimensions
    if len(roi) != 4:
        raise ValueError(
            "ROI must be a tuple of four integers (y_start, y_end, x_start, x_end).")

    y_start, y_end, x_start, x_end = roi

    # Handle zero indices for y_end and x_end (full extent)
    if y_end == 0:
        y_end = imgs.shape[1]
    if x_end == 0:
        x_end = imgs.shape[2]

    # Handle negative indices for y_end and x_end
    if y_end < 0:
        y_end = imgs.shape[1] + y_end
    if x_end < 0:
        x_end = imgs.shape[2] + x_end

    # Validate ROI bounds
    if not (0 <= y_start < imgs.shape[1] and 0 <= y_end <= imgs.shape[1] and
            0 <= x_start < imgs.shape[2] and 0 <= x_end <= imgs.shape[2]):
        raise ValueError(
            "ROI coordinates are out of bounds of the image dimensions.")

    # Crop the images
    return imgs[:, y_start:y_end, x_start:x_end] if not was_2d else imgs[0, y_start:y_end, x_start:x_end]


def split_n_shift(img: np.ndarray, n_windows: tuple[int, int], overlap: float = 0, shift: tuple[int, int] | np.ndarray = (0, 0), shift_mode: str = 'before', plot: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a 2D image array (y, x) into (overlapping) windows,
    with automatic window size adjustments for shifted images.

    Args:
        img (np.ndarray): 2D array of image values (y, x).
        n_windows (tuple[int, int]): Number of windows in (y, x) direction.
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
    n_y, n_x = n_windows

    # Handle both uniform and non-uniform shifts
    shift_array = np.asarray(shift)

    # Handle NaN values before casting to int
    if np.any(np.isnan(shift_array)):
        # Replace NaN values with 0 (no shift for invalid windows)
        shift_array = np.nan_to_num(shift_array, nan=0.0)

    # Now safely cast to int
    shift_array = shift_array.astype(int)

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
    window_y0_indices = np.linspace(0, img_h - split_img_h, num=n_y, dtype=int)
    window_x0_indices = np.linspace(0, img_w - split_img_w, num=n_x, dtype=int)
    pos_grid = np.stack(np.meshgrid(
        window_y0_indices, window_x0_indices, indexing="ij"), axis=-1)

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
    windows = np.zeros((n_y, n_x, win_h, win_w), dtype=img.dtype)
    for i, y in enumerate(window_y0_indices):
        for j, x in enumerate(window_x0_indices):

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
                windows[i, j] = np.pad(win_crop, ((pad_y_top, pad_y_bottom),
                                                  (pad_x_left, pad_x_right)),
                                       mode='constant', constant_values=0)
            else:
                windows[i, j] = win_crop

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

    return windows, win_pos
