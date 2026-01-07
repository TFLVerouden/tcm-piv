import os
import pickle

import cv2 as cv
import numpy as np
from natsort import natsorted
from scipy import signal as sig
from scipy.interpolate import make_smoothing_spline
from skimage.feature import peak_local_max
from tqdm import tqdm
import matplotlib.pyplot as plt


def downsample(img, factor):
    """Downsample a 2D image by summing non-overlapping blocks
     of size (block_size, block_size)."""
    h, w = img.shape
    assert h % factor == 0 and w % factor == 0, \
        "Image dimensions must be divisible by block_size"
    return img.reshape(h // factor, factor,
                       w // factor, factor).sum(axis=(1, 3))


def split_image(imgs, nr_windows, overlap=0, shift=(0, 0), shift_mode='before',
                plot=False):
    """
    Split a 3D image array (n_img, y, x) into (overlapping) windows, with optional edge cut-off for shifted images.

    Parameters:
        imgs (np.ndarray): 3D array of image values (image_index, y, x).
        nr_windows (tuple): Number of windows in (y, x) direction.
        overlap (float): Fractional overlap between windows (0 = no overlap).
        shift (tuple): (dx, dy) shift in pixels - can be (0, 0).
        shift_mode (str): 'before' or 'after' - which frame is considered?
        plot (bool): If True, plot the windows on the first image.

    Returns:
        windows (np.ndarray): 5D array of image windows
            (image_index, window_y_idx, window_x_idx, y, x).
        centres (np.ndarray): 3D array of window centres
            (window_y_idx, window_x_idx, 2).
    """
    n_img, img_y, img_x = imgs.shape
    n_y, n_x = nr_windows
    dy, dx = shift

    # Calculate window size including overlap
    size_y = min(int(img_y // n_y * (1 + overlap)), img_y)
    size_x = min(int(img_x // n_x * (1 + overlap)), img_x)

    # Get the top-left corner of each window
    y_indices = np.linspace(0, img_y - size_y, num=n_y, dtype=int)
    x_indices = np.linspace(0, img_x - size_x, num=n_x, dtype=int)

    # Create grid of window coordinates
    grid = np.stack(np.meshgrid(y_indices, x_indices, indexing="ij"), axis=-1)

    # Compute centres (window_y_idx, window_x_idx, 2)
    centres = np.stack((grid[:, :, 0] + size_y / 2,
                        grid[:, :, 1] + size_x / 2), axis=-1)

    # Determine cut-off direction: +1 for 'before', -1 for 'after'
    mode_sign = 1 if shift_mode == 'before' else -1

    # Calculate cut-off for each direction
    cut_y0 = max(0,  mode_sign * dy); cut_y1 = max(0, -mode_sign * dy)
    cut_x0 = max(0,  mode_sign * dx); cut_x1 = max(0, -mode_sign * dx)

    # Show windows and centres on the first image if requested
    if plot:
        fig, ax = plt.subplots()
        ax.imshow(imgs[0].astype(float) / imgs[0].max() * 255, cmap='gray')
    # Pre-allocate and fill the windows
    windows = np.empty((n_img, n_y, n_x,
                        size_y - abs(dy), size_x - abs(dx)), dtype=imgs.dtype)
    for img_idx in range(n_img):
        for i, y in enumerate(y_indices):
            for j, x in enumerate(x_indices):
                y0 = y + cut_y0; y1 = y + size_y - cut_y1
                x0 = x + cut_x0; x1 = x + size_x - cut_x1
                windows[img_idx, i, j] = imgs[img_idx, y0:y1, x0:x1]

                if plot:
                    color = ['orange', 'blue'][(i + j) % 2]
                    rect = plt.Rectangle((x + cut_x0, y + cut_y0),
                                            x + size_x - cut_x1 - (x + cut_x0),
                                            y + size_y - cut_y1 - (y + cut_y0),
                                            edgecolor=color, facecolor='none', linewidth=1.5)
                    ax.add_patch(rect)
                    ax.scatter(centres[i, j, 1], centres[i, j, 0], c=color, marker='x', s=40)

    # Finish plot
    if plot:
        plt.xlim(-20, img_x + 20)
        plt.ylim(-20, img_y + 20)  # Invert y-axis for image coordinates??
        plt.show()

    return windows, centres


def find_peaks(corr_map, num_peaks=1, min_distance=5):
    """
    Find peaks in a correlation map.

    Parameters:
        corr_map (np.ndarray): 2D array of correlation values.
        num_peaks (int): Number of peaks to find.
        min_distance (int): Minimum distance between peaks in pixels.

    Returns:
        peaks (np.ndarray): Array of peak coordinates shaped (num_peaks, 2)
        intensities (np.ndarray): Intensities of the found peaks.
    """

    # Todo: peaks should not be at the edge of the correlation map
    if num_peaks == 1:
        # Find the single peak
        peaks = np.argwhere(np.amax(corr_map) == corr_map)
    else:
        # Find multiple peaks using peak_local_max
        peaks = peak_local_max(corr_map, min_distance=min_distance, num_peaks=num_peaks)

    return peaks, corr_map[peaks[:, 0], peaks[:, 1]]


def three_point_gauss(array):
    # Fit a Gaussian to three points
    return (0.5 * (np.log(array[0]) - np.log(array[2])) /
            ((np.log(array[0])) + np.log(array[2]) - 2 * np.log(array[1])))


def subpixel(corr_map, peak):
    # Use a Gaussian fit to refine the peak coordinates
    y_corr = three_point_gauss(corr_map[peak[0]-1:peak[0]+2, peak[1]])
    x_corr = three_point_gauss(corr_map[peak[0], peak[1]-1:peak[1]+2])

    # Add subpixel correction to the peak coordinates
    return peak.astype(np.float64) + np.array([y_corr, x_corr])


def remove_outliers(coords, y_max, x_max, strip=True):
    """
    Remove outliers:
    - For x < 0: keep only points inside a semi-circle of radius y_max centered at (0,0)
    - For x >= 0: keep only points inside a rectangle [-0.5, x_max] x [-y_max, y_max]
    """

    # Coords might be an 3D array. Reshape it to 2D for processing
    orig_shape = coords.shape
    coords = coords.reshape(-1, coords.shape[-1])

    # Set all non-valid coordinates to NaN
    mask = (((coords[:, 1] < 0) &
             (coords[:, 1]**2 + coords[:, 0]**2 <= y_max**2))
            | ((coords[:, 1] >= 0) & (coords[:, 1] <= x_max)
               & (np.abs(coords[:, 0]) <= y_max)))
    coords[~mask] = np.array([np.nan, np.nan])

    # Reshape back to original shape
    coords = coords.reshape(orig_shape)

    # If needed, reduce the array to 2D by taking only the first non-NaN coordinate
    if strip and coords.ndim > 2:
        coords_stripped = np.full([coords.shape[0], 2], np.nan, dtype=np.float64)
        for i in range(coords.shape[0]):
            for j in range(coords.shape[1]):
                if ~np.any(coords[i, j, :] == np.nan):
                    # If there are non NaNs, save these coordinates
                    coords_stripped[i, :] = coords[i, j, :]
                    break
                elif j == coords.shape[1] - 1:
                    # If all coordinates are NaN, set to NaN
                    coords_stripped[i, :] = np.array([np.nan, np.nan])
        coords = coords_stripped

    return coords

if __name__ == "__main__":
    # Set variables
    test_mode = True
    data_path = '/Volumes/Data/Data/250623 PIV/250624_1333_80ms_whand'
    # data_path = ('/Users/tommieverouden/PycharmProjects/cough-machine-control/piv/'
    #         'test_pair')
    cal_path = ('/Users/tommieverouden/PycharmProjects/cough-machine-control/piv/'
                'calibration/250624_calibration_PIV_500micron_res_std.pkl')

    # In the current directory, create a folder named the same as the final part of the data_path
    proc_path = os.path.join(os.getcwd(), 'processed', os.path.basename(data_path))
    if not os.path.exists(proc_path) and not test_mode:
        os.makedirs(proc_path)

    frame_nrs = [930, 931] if test_mode else list(range(1, 6000))
    dt = 1/40000 # s
    v_max = [15, 150] # m/s
    downs_fac = 4  # First pass downsampling
    num_peaks = 10  # Number of peaks to find in first pass correlation map

    # Read calibration data
    res_avg, _ = np.load(cal_path)

    # Convert max velocities to max displacements in px
    d_max = np.array(v_max) * dt / res_avg  # m/s -> px/frame


    # FIRST PASS: Full frame correlation
    # Shortcut: if a disp1.npz file already exists, load it
    disp1_path = os.path.join(proc_path, 'disp1.npz')
    if os.path.exists(disp1_path) and not test_mode:
        with np.load(disp1_path, allow_pickle=True) as data:
            disp1 = data['disp1']
            disp1_unf = data['disp1_unf']
            time = data['time']
        print("Loaded existing disp1.npz file.")
    else:

        # List all images in folder; filter for .tif files; sort; get specific frames
        files = natsorted([f for f in os.listdir(data_path) if f.endswith('.tif')])
        files = [f for f in files if any(f.endswith(f"{nr:05d}.tif") for nr in
                                         frame_nrs) and not f.startswith('.')]

        # Import images into 3D numpy array (image_index, y, x)
        imgs = np.array([cv.imread(os.path.join(data_path, f), cv.IMREAD_GRAYSCALE)
                         for f in tqdm(files, desc='Reading images')],
                        dtype=np.uint64)

        # TODO: Pre-process images (background subtraction? thresholding?
        #  binarisation to reduce relative influence of bright particles?
        #  low-pass filter to remove camera noise?
        #  mind increase in measurement uncertainty -> PIV book page 140)

        n_frames = len(imgs) - 1

        # Pre-allocate array for all peaks: (n_frames, num_peaks, 2) [vy, vx]
        disp1 = np.full((n_frames, num_peaks, 2), np.nan)

        # Define time arrays beforehand
        time = np.linspace((frame_nrs[0] - 1) * dt, (frame_nrs[0] - 1 + n_frames - 1) * dt, n_frames)

        for i in tqdm(range(n_frames), desc='First pass'):
            img1 = downsample(imgs[i + 1], downs_fac)
            img0 = downsample(imgs[i], downs_fac)
            corr_map = sig.correlate(img1, img0, method='fft')
            peaks, int_unf = find_peaks(corr_map, num_peaks=num_peaks, min_distance=5)

            # Calculate velocities for all peaks
            disp1[i, :, :] = (peaks - np.array(corr_map.shape) // 2) * downs_fac  # shape (n_found, 2)

        # Save unfiltered displacements
        disp1_unf = disp1.copy()

        # Outlier removal
        # TODO: Do something with the intensities of the peaks?
        disp1 = remove_outliers(disp1, y_max=d_max[0], x_max=d_max[1], strip=True)

        # Save the displacements to a file
        if not test_mode:
            np.savez(os.path.join(proc_path, 'disp1'), time=time, disp1=disp1,
                     disp1_unf=disp1_unf, int_unf=int_unf)

    # # Interpolate data to smooth out the x_displacement in time
    # disp1_spl = make_smoothing_spline(time[~np.isnan(disp1[:, 1])],
    #                                   disp1[~np.isnan(disp1[:, 1]), 1], lam=5e-7)
    # disp1_spl = disp1_spl(time).astype(int)
    #
    # # Calculate velocities for plot
    # vel1_unf = disp1_unf * res_avg / dt
    # vel1 = disp1 * res_avg / dt
    # vel1x_spl = disp1_spl * res_avg / dt
    #
    # # Scatter plot vx(t)
    # plt.figure()
    # plt.scatter(np.tile(1000*time[:, None], (1, num_peaks)), vel1_unf[..., 1],
    #             c='gray', s=2, label='Other peaks')
    # plt.scatter(1000*time, vel1_unf[:, 0, 1], c='blue', s=10,
    #             label='Most prominent peak')
    # plt.scatter(1000*time, vel1[:, 1], c='orange', s=4,
    #             label='After outlier removal')
    # plt.plot(1000*time, vel1x_spl, label='Displacement to be used\n in 2nd pass (smoothed)', color='red')
    # plt.ylim([-15, 150])
    # plt.xlabel('Time (ms)')
    # plt.ylabel('vx (m/s)')
    # plt.legend(loc='upper right', fontsize='small', framealpha=1)
    #
    # # Save plot as pdf
    # if ~test_mode:
    #     plt.savefig(os.path.join(proc_path, 'disp1_vx_t.pdf'), bbox_inches='tight')
    # plt.show()

    # Split image nr 930 into windows
    windows1, centres1 = split_image(imgs, (4,1), overlap=0.2, shift=(0, 20), shift_mode='after',
                plot=True)
    windows0, centres0 = split_image(imgs, (4,1), overlap=0.2, shift=(0, 20), shift_mode='before',
                plot=True)
    print()

    # map1 = sig.correlate(downsample(imgs[1], factor=8), downsample(imgs[0], factor=8), method='fft')
    # peaks, _ = find_peaks(map1, num_peaks=5, min_distance=5)
    #
    # # Todo: check this list of peak with previous and next frame (see step 3 in PIV book page 148)
    # # If none match, interpolate between the two frames. For now, just take the first peak.
    #
    # disp1 = peaks[0] - np.array(map1.shape) // 2
    # print(disp1 * 8 * res_avg / dt)

    # # Split images into overlapping windows
    # windows, centres = split_image(imgs, nr_windows=(16, 1), overlap=0.5)
    #
    # # # Plot the windows and centres on top of the first image
    # # plt.imshow(imgs[0], cmap='gray')
    # # for i, window in enumerate(windows):
    # #     y, x = centres[i]
    # #     rect = plt.Rectangle((x - window.shape[1] / 2, y - window.shape[0] / 2),
    # #                          window.shape[1], window.shape[0], linewidth=1,
    # #                          edgecolor='r', facecolor='none')
    # #     plt.gca().add_patch(rect)
    # #     plt.plot(x, y, 'ro')  # Plot the centre
    # # plt.show()
    #
    # # Cycle through all windows in one specific image and correlate them with the corresponding windows in the other image
    # maps = np.array([[sig.correlate(window[1], window[0], method='fft')
    #          for window in zip(windows[0], windows[1])]])
    #
    # # TODO: Any processing of the correlation map happens here (i.e. blacking out all pixels outside of a positive semi-circle)
    #
    # peak, int = find_peaks(maps[0, 7, 0])
    # print("Peak coordinates:", peak)
    # print("Peak intensity:", int)
    # peak = subpixel(maps[0, 7, 0], peak[0])
    # print("Subpixel peak coordinates:", peak)
    #
    # # Get displacement vector from the peak coordinates
    # displacement_vector = peak - np.array(maps.shape[3:]) // 2
    #
    # print(displacement_vector * res_avg / dt)
