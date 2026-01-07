import numpy as np
import matplotlib.pyplot as plt


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
    # print(f"Window size: {size_y} x {size_x}, overlap: {overlap:.2f}")

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

    # Debug info
    # print(f"Original window size: {size_y} x {size_x}")
    # print(f"Max shifts: dy={max_abs_dy}, dx={max_abs_dx}")
    # print(f"Target uniform size: {uniform_size_y} x {uniform_size_x}")

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
        # plt.show()

    return wins, win_pos


if __name__ == "__main__":

    overlap = 0

    # Test 1
    n_wins = (8, 1)
    shift_nonuni = np.array([
        [[0, 0]],
        [[0, 5]],
        [[5, 10]],
        [[0, 10]],
        [[0, 15]],
        [[5, 20]],
        [[0, 20]],
        [[0, 25]]
    ])

    # # Test 2
    # n_wins = (5, 3)
    # shift_nonuni = np.array([
    #     [[0, 0], [0, 5], [5, 10]],
    #     [[0, 10], [0, 15], [5, 20]],
    #     [[0, 10], [0, 15], [5, 20]],
    #     [[10, 20], [5, 10], [0, 0]],
    #     [[5, 10], [10, 30], [5, 20]]
    # ])

    # Uniform test
    shift_uni = (5, 20)

    # Generate test images
    img0 = np.zeros((832, 384), dtype=np.uint8)
    img1 = np.zeros((832, 384), dtype=np.uint8)

    # Add several small squares at different positions
    for i in range(n_wins[0]):
        for j in range(n_wins[1]):
            # Calculate window position
            y_start = int(i * 832 // n_wins[0]) + 10
            x_start = int(j * 384 // n_wins[1]) + 10

            # Add square in original image
            img0[y_start:y_start+20, x_start:x_start+20] = 255

            # Add shifted square in second image
            dy, dx = shift_nonuni[i, j]
            img1[y_start+dy:y_start+20+dy,
                        x_start+dx:x_start+20+dx] = 255

    # Test uniform shift (backward compatibility)
    print("Testing uniform shift...")
    wnd0, _ = split_n_shift(img0, n_wins, shift=shift_uni,
                            overlap=overlap, shift_mode='before', plot=True)
    wnd1, _ = split_n_shift(img1, n_wins, shift=shift_uni,
                            overlap=overlap, shift_mode='after', plot=True)

    # Test non-uniform shift
    print("Testing non-uniform shift...")
    wnd0_nonuni, _ = split_n_shift(img0, n_wins, shift=shift_nonuni,
                                   overlap=overlap, shift_mode='before', plot=True)
    wnd1_nonuni, _ = split_n_shift(img1, n_wins, shift=shift_nonuni,
                                   overlap=overlap, shift_mode='after', plot=True)

    # Print information about the windows
    print(f"Uniform shift windows shape: {wnd0.shape}")
    print(f"Non-uniform shift windows shape: {wnd0_nonuni.shape}")

    # Print sample window sizes for non-uniform case (should all be uniform now)
    print("Sample window sizes (non-uniform, now padded to uniform):")
    for i in range(min(2, wnd0_nonuni.shape[0])):
        for j in range(min(2, wnd0_nonuni.shape[1])):
            window = wnd0_nonuni[i, j]
            shift_val = shift_nonuni[i, j]
            print(f"  Window [{i},{j}] with shift {shift_val}: {window.shape}")

    # Show that all windows now have the same size
    print(
        f"All windows are uniform size: {wnd0_nonuni.shape[2:]} (height x width)")

    # Verify that features are positioned correctly relative to their shifts
    print("\nFeature preservation check:")
    print("- Features should maintain their relative positions within windows")
    print("- Zero padding fills the areas that would be outside the original crop")

    # Quick check: show a few sample windows to verify feature positioning
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.suptitle('Sample Windows - Before (top) and After (bottom) shift')

    for idx in range(min(4, n_wins[0])):
        # Show before and after windows for the same position
        axes[0, idx].imshow(wnd0_nonuni[idx, 0], cmap='gray')
        axes[0, idx].set_title(f'Before\nshift={shift_nonuni[idx, 0]}')
        axes[0, idx].axis('off')

        axes[1, idx].imshow(wnd1_nonuni[idx, 0], cmap='gray')
        axes[1, idx].set_title(f'After\nshift={shift_nonuni[idx, 0]}')
        axes[1, idx].axis('off')

    plt.tight_layout()

    plt.show()
