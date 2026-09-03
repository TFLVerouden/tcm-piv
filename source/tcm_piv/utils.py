"""
Utility functions for PIV analysis.

This module contains general-purpose utility functions used throughout
the PIV analysis workflow, including coordinate transformations,
time calculations, and displacement-to-shift conversions.
"""

import numpy as np


def double_bits(arr: np.ndarray) -> np.ndarray:
    """
    Double the number of bits in an array.

    This function takes an input array and returns a new array with double the
    number of bits, effectively converting the data type to a larger integer
    type. For example, uint8 becomes uint16, int16 becomes int32, etc.

    Args:
        arr (np.ndarray): Input array of integer type.

    Returns:
        np.ndarray: New array with double the number of bits.
    """

    if arr.dtype == np.uint8:
        return arr.astype(np.uint16)
    elif arr.dtype == np.uint16:
        return arr.astype(np.uint32)
    elif arr.dtype == np.uint32:
        return arr.astype(np.uint64)
    else:
        return arr


def reduce_bits(arr: np.ndarray) -> np.ndarray:
    """
    Based on the maximum value in the array, reduce the number of bits to the
    smallest possible integer type that can hold the data.
    """
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError("Input array must be of integer type.")

    max_val = np.max(arr)
    if max_val == 0:
        return arr.astype(np.uint8)

    # Find the smallest integer type that can hold the maximum value
    for dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
        if np.iinfo(dtype).max >= max_val:
            return arr.astype(dtype)

    raise ValueError("Cannot reduce bits to a smaller integer type.")


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


def disp2shift(n_windows: tuple[int, int], displacements: np.ndarray) -> np.ndarray:
    """
    Distribute displacement values over a larger number of windows.

    This function takes displacement values from a smaller window grid and 
    distributes them to a larger window grid by replicating values across
    multiple windows.

    Args:
        n_windows (tuple[int, int]): Target number of windows (n_y, n_x)
        displacements (np.ndarray): Displacement array with shape (n_frames, n_y_source, n_x_source, 2)

    Returns:
        np.ndarray: Shift array with shape (n_frames, n_y_target, n_x_target, 2)
                   compatible with split_n_shift function
    """
    n_y_target, n_x_target = n_windows

    # Validate input shape
    if displacements.ndim != 4 or displacements.shape[3] != 2:
        raise ValueError(
            "Displacement array must have 4 dimensions (n_frames, n_y, n_x, 2), "
            f"got {displacements.shape}"
        )

    n_frames, n_y_source, n_x_source, _ = displacements.shape

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
            shifts[:, y0:y1, x0:x1, :] = displacements[:, i:i+1, j:j+1, :]

    return shifts


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


def vel2flow(
    vel: np.ndarray,
    d: float,
    image_width_m: float,
    image_height_m: float,
    *,
    flow_direction: str = "x",
) -> np.ndarray:
    """
    Convert velocity profile to volumetric flow rate.

    Integrates the velocity field over the cross-sectional area to calculate
    the total volumetric flow rate for each frame. If ANY velocity value is NaN,
    the corresponding flow rate will be NaN.

    Args:
        vel (np.ndarray): Velocity array with shape (n_frames, n_y, n_x, 2)
                         where vel[..., 0] is vy and vel[..., 1] is vx
        d (float): Depth of the measurement field in meters (out-of-plane dimension)
        image_width_m (float): Full image width in meters.
        image_height_m (float): Full image height in meters.
        flow_direction (str): Which in-plane component represents flow direction.
            Allowed: "x" (use vx) or "y" (use vy).

    Returns:
        np.ndarray: Flow rate array with shape (n_frames,) in m³/s
                   Multiply by 1000 to get L/s
    """
    flow_dir = str(flow_direction).strip().lower()
    if flow_dir == "x":
        # Take the first column of windows, and the velocity in the x direction
        vel_component = vel[:, :, 0, 1]

        # The cross-stream width is the image height
        cross_stream_width_m = float(image_height_m)

        # Take the first windows in the x-direction as the flow is assumed uniform across the width.
    elif flow_dir == "y":
        vel_component = vel[:, 0, :, 0]
        cross_stream_width_m = float(image_width_m)
    else:
        raise ValueError("flow_direction must be 'x' or 'y'")

    # Calculate the mean velocity across the cross-stream direction for each frame
    # Always take the first window in the cross-stream direction, as the flow is assumed uniform across the depth.
    # mean over y and x windows
    v_mean = np.nanmean(vel_component, axis=1)

    # Keep behavior predictable: if any window is NaN for a frame,
    # mark that frame's flow as NaN.
    frame_has_nan = np.any(np.isnan(vel_component), axis=1)
    v_mean = v_mean.astype(float, copy=False)
    v_mean[frame_has_nan] = np.nan

    flow_rate = v_mean * float(d) * cross_stream_width_m  # m^3/s

    return flow_rate
