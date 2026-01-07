import numpy as np
from matplotlib import pyplot as plt
from natsort import natsorted
import cv2 as cv
from numba import jit, njit
from tqdm import tqdm
from tqdm import trange
import os
import warnings
from scipy.optimize import curve_fit
from scipy import signal as sig


def read_image_directory(directory, prefix=None, image_type='png',
                         file_range=None, timing=False):
    """
    Read all images in a directory and store them in a 3D array.

    Parameters
    ----------
    directory : str
        Path to the directory containing the images.
    prefix : str
        Prefix of the image files to read.
    image_type : str
        Type of the image files to read.
    file_range : int | tuple
        Number of images to read or a tuple of start and stop indices.
        Default: None.
    timing : bool
        Whether to show a progress bar.

    Returns
    -------
    images : np.array
        Array (i, y, x) containing the images.
    """

    # Get a list of files in the directory
    files = os.listdir(directory)

    # If a prefix is specified, filter the list of files
    if prefix is not None:
        files = [f for f in files if f.startswith(prefix)]

    # If a type is specified, filter the list of files
    if image_type is not None:
        files = [f for f in files if f.endswith(image_type)]

    # Sort the files
    files = natsorted(files)

    # If indices is an integer, treat it as the stop point
    if isinstance(file_range, int):
        files = files[:(file_range + 1)]

    # If it is a tuple, treat it as start and stop points
    elif isinstance(file_range, tuple):
        files = files[(file_range[0]):(file_range[1] + 1)]

    # Read the images and store them in a 3D array
    images = np.array([cv.imread(os.path.join(directory, f),
                                 cv.IMREAD_GRAYSCALE) for f in
                       tqdm(files, desc='Reading images', disable=not timing)],
                      dtype=np.uint64)

    return images


def subtract_background(images, background, timing=False):
    """
    Subtract a background image from a set of images.

    PARAMETERS:
        images (np.array): Images [c, y, x].
        background (np.array): Background image [y, x].

    RETURNS:
        images (np.array): Images with background subtracted [c, y, x].
    """

    # Check whether the images and background have the same shape
    if images.shape[1:] != background.shape:
        # Error
        raise ValueError(
                'The images and background do not have the same shape.')

    # Subtract the background from the images
    for c in trange(images.shape[0], desc='Subtracting background',
                    disable=not timing):
        images[c] = images[c] - background

    # Set any integer overflowed values to zero
    images[images > 255] = 0

    return images


def displacement_1d(correlation, axis=1, max_disp=None, ignore_disp=0,
                    ignore_only_if_mult=True, subpixel_method=None, plot=False):
    # # Cross-correlate the images
    # correlation = sig.correlate(image1, image0)
    # Idea: Possibly better to calculate manually the correlation along 1 axis

    if axis == 1:
        # Get a slice of the correlation image
        correlation_slice = correlation[correlation.shape[0] // 2, :]

        # If a maximum displacement is specified, cut the slice from the center
        if max_disp is not None:
            correlation_slice = correlation_slice[
                                correlation.shape[1] // 2 - max_disp:
                                correlation.shape[1] // 2 + max_disp]

        # Define the image center
        center = correlation_slice.shape[0] // 2
    else:
        raise NotImplementedError('Only axis=1 is implemented')

    # Get local maxima in the correlation slice
    prominence = 0.5 * np.std(correlation_slice)
    peaks = sig.find_peaks(correlation_slice, prominence=prominence, width=1)[0]

    # Filter out peaks that are too far from the image center
    if max_disp is not None:
        peaks = peaks[np.abs(peaks - center) < max_disp]

    # If a displacement to be ignored is specified, and ignore is set to always,
    # or there are multiple peaks...
    if ignore_disp is not None and (len(peaks) > 1 or not ignore_only_if_mult):
        # Remove the specified peak
        peaks = peaks[peaks != ignore_disp + center]

        # Sort the peaks by correlation value in decreasing order
        peaks = peaks[np.argsort(correlation_slice[peaks])[::-1]]

    # If there are no peaks, no PIV can be done
    if len(peaks) == 0:
        return np.nan

    # Use the brightest peak as the displacement
    peak = peaks[0]

    # If subpixel resolution is requested, calculate it
    if subpixel_method is not None:
        # Calculate the subpixel displacement
        correction = subpixel_correction(correlation_slice, peak,
                                         method=subpixel_method)

        # Add the subpixel displacement to the integer displacement
        peak = peak + correction

    # Subtract the image center to get the displacement
    displacement = peak - center

    # Plot the correlation slice and the peak
    if plot:
        if max_disp is None:
            max_disp = correlation.shape[1] // 2

        x = np.arange(correlation.shape[1]) - correlation.shape[1] // 2

        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(x, correlation[correlation.shape[0] // 2, :], '-')
        ax.axvline(displacement, color='r')
        ax.set_xlim(-max_disp, max_disp)
        ax.set_xlabel('Displacement [px]')
        ax.set_ylabel('Correlation')
        plt.show()

    return displacement


def displacement_2d(correlation, max_disp=None,
                    subpixel_method='gauss_neighbor', plot=False):
    # Plot the correlation map
    if plot:
        extent = [-correlation.shape[1] // 2, correlation.shape[1] // 2,
                  -correlation.shape[0] // 2, correlation.shape[0] // 2]

        fig, ax = plt.subplots()
        ax.imshow(correlation, extent=extent)
        ax.set_xlabel('dx [px]')
        ax.set_ylabel('dy [px]')
        plt.show()

    # If all values in the correlation array are zero...
    if np.all(correlation == 0):
        # Return a nan displacement
        return np.array([np.nan, np.nan])

    # Set all values outside a circle with radius max_displ to zero
    if max_disp is not None:
        # Set max_disp to the maximum possible displacement if it is too large
        max_disp = min(max_disp, correlation.shape[0] // 2 - 1,
                       correlation.shape[1] // 2 - 1)

        # Create a grid of distances from the center
        x, y = np.meshgrid(
                np.arange(correlation.shape[1]) - correlation.shape[1] // 2,
                np.arange(correlation.shape[0]) - correlation.shape[0] // 2)
        r = np.sqrt(x ** 2 + y ** 2)

        # Set all values outside the circle to zero
        correlation[r > max_disp] = 0

    # Get the pixel with maximum brightness
    peaks = np.argwhere(np.amax(correlation) == correlation)

    # If multiple equal maxima were found...
    if len(peaks) > 1:

        # Use the peak with the largest sum of its neighbours
        peak = peaks[np.argmax([np.sum(correlation[p[0] - 1:p[0] + 2,
                                       p[1] - 1:p[1] + 2]) for p in peaks])]

    else:
        # Take the first value if only one peak was found
        peak = peaks[0]

    # If the subpixel option was set...
    if subpixel_method is not None:
        # Try calculating the subpixel correction
        try:
            # Get slices along both axes
            x_slice = correlation[peak[0], :]
            y_slice = correlation[:, peak[1]]

            # Calculate the subpixel displacement
            correction = np.array([
                subpixel_correction(x_slice, peak[1], method=subpixel_method),
                subpixel_correction(y_slice, peak[0], method=subpixel_method)])

            # If the displacement is within the floating point error...
            if np.abs(np.linalg.norm(correction)) < 1e-6:
                # Return a nan displacement
                return np.array([np.nan, np.nan])

            # Add the subpixel displacement to the integer displacement
            peak = peak + correction

        # If the subpixel calculation failed, return a nan displacement
        except (ValueError, FloatingPointError):
            return np.array([np.nan, np.nan])

        # Add the subpixel displacement to the integer displacement
        peak = peak + correction

    # Subtract the image center to get the displacement
    center = (np.array(correlation.shape - np.ones_like((1, 2))) / 2)
    displacement = peak - center

    return displacement


def shift_displaced_image(images, displacement, axis=1):
    # Make a copy of the images_shifted
    images_shifted = np.copy(images)

    # If the displacement is non-zero...
    if displacement != 0:

        # Shift the image
        if axis == 1:
            images_shifted[1, :, :] = np.roll(images_shifted[1, :, :],
                                              -displacement)

            # Zero the pixels that were shifted out of the image
            # ISSUE: this introduces a high-frequency component at the edge
            if displacement > 0:
                images_shifted[1, :, -displacement:] = 0
            else:
                images_shifted[1, :, :-displacement] = 0
        else:
            raise NotImplementedError('Only axis=1 is implemented')

    return images_shifted


def subpixel_correction(array, peak_index, method='gauss_neighbor'):
    """
    """

    # Three-point offset calculation from the lecture
    if method == 'gauss_neighbor':

        # If the array has only equal values, raise error
        if np.all(array == array[0]):
            raise ValueError('All values in the array are equal')

        # Raise error if numpy encounters an exception
        with np.errstate(divide='raise', invalid='raise'):

            # Get the neighbouring pixels
            neighbors = array[(peak_index - 1):(peak_index + 2)]

            # Calculate the three-point Gaussian correction
            correction = (0.5 * (np.log(neighbors[0]) - np.log(neighbors[2]))
                          / ((np.log(neighbors[0])) + np.log(neighbors[2]) -
                             2 * np.log(neighbors[1])))
    else:
        raise ValueError('Invalid method')

    return correction


def plot_flow_field(displacements, window_centers, background=None,
                    margins=np.array([0, 0, 0, 0]), arrow_color='k',
                    arrow_scale=1, center_arrows=True,
                    zero_displ_thr=0, axis_labels=['x', 'y'],
                    highlight_radius_range=[np.nan, np.nan],
                    highlight_angle_range=[np.nan, np.nan],
                    highlight_color='b', calib_dist=None, right_axis_scale = None,
                    units='px', title='Flow field', time_stamp=None, timing=False):
    """
    Plot the flow field with displacements as arrows.

    This function takes in the same keyword arguments as plot_displacements,
    but ignores some of them.

    PARAMETERS:
        displacements (np.array): Displacement vectors [j, i, y/x].
        window_centers (np.array): Coordinates of the windows [j, i, y/x].
        background (np.array): Background image to plot the flow field on.
        arrow_color (str): Color of the arrows.
        arrow_scale (float): Scale of the arrows.
        zero_displ_thr (float): Threshold for displacements to be plotted
            as dots.
        highlight_radius_range (list): Range of magnitudes to highlight.
        highlight_angle_range (list): Range of angles to highlight.
        highlight_color (str): Color of the highlighted arrows and dots.
        calib_dist (float): Calibration distance.
        units (str): Units of the calibration distance.
        title (str): Title of the plot.
        timing (bool): Whether to show a progress bar.

    RETURNS:
        fig (plt.figure): Figure object.
        ax (plt.axis): Axis object.
    """

    # Plot all displacement vectors at the center of each window
    fig, ax = plt.subplots()

    # Set the image extent and center the windows if a background is supplied
    if background is not None:
        extent = np.array([-background.shape[1] / 2, background.shape[1] / 2,
                           -background.shape[0] / 2, background.shape[0] / 2])
        window_centers = window_centers - np.array([background.shape[0] / 2,
                                                    background.shape[1] / 2])

    # If margins are specified, shift the window positions and image extent
    if np.any(margins):
        window_centers = window_centers + np.array([margins[0], margins[2]])

        if background is not None:

            extent = extent - np.array([margins[2], margins[2],
                                        margins[0], margins[0]])

    # If a calibration distance was specified, calibrate all values
    if calib_dist is not None:
        displacements = displacements * calib_dist / arrow_scale
        window_centers = window_centers * calib_dist / arrow_scale
        extent = extent * calib_dist / arrow_scale
        zero_displ_thr = zero_displ_thr * calib_dist / arrow_scale

    # If a background image is supplied, add it to the plot
    if background is not None:
        ax.imshow(background, cmap='gray', extent=extent)

    # Show a grid with the outline of each window and an arrow in the centre
    # indicating the displacement
    arrow_param = calib_dist / arrow_scale if calib_dist is not None else 1

    # Get a list of indices that should be coloured
    highlight = filter_displacements(displacements,
                                     radius_range=highlight_radius_range,
                                     angle_range=highlight_angle_range)

    # Get a list of indices that are below the zero-threshold
    zero_displ = np.nan_to_num(
            np.linalg.norm(displacements, axis=2)) <= zero_displ_thr

    # Plot the indices below the threshold as dots
    ax.scatter(window_centers[zero_displ & highlight, 1],
               window_centers[zero_displ & highlight, 0],
               c=highlight_color, marker='.', s=arrow_scale)
    ax.scatter(window_centers[zero_displ & ~highlight, 1],
               window_centers[zero_displ & ~highlight, 0],
               c=arrow_color, marker='.', s=arrow_scale)

    # Plot the flow field window by window
    for j in trange(window_centers.shape[0], desc='Plotting arrows',
                    disable=not timing):
        for i in range(window_centers.shape[1]):

            # If the displacement is above the zero-threshold, plot an arrow
            if np.linalg.norm(displacements[j, i]) > zero_displ_thr:

                # If the displacement should be highlighted, set the color
                color = highlight_color if highlight[j, i] else arrow_color

                # Calculate the start of the arrow
                if center_arrows:
                    arrow_start = np.array(
                            [window_centers[j, i][0] -
                             arrow_scale * 0.5 * displacements[j, i][0],
                             window_centers[j, i][1] -
                             arrow_scale * 0.5 * displacements[j, i][1]])
                else:
                    arrow_start = window_centers[j, i]

                # Plot the arrow
                ax.arrow(arrow_start[1], arrow_start[0],
                         arrow_scale * displacements[j, i][1],
                         arrow_scale * displacements[j, i][0],
                         width=1.5 * arrow_param,
                         head_width=10 * arrow_param,
                         head_length=7 * arrow_param,
                         fc=color, ec=color, lw=1)

    # Aspect ratio should be 1
    ax.set_aspect('equal')

    # Add the time stamp to the top left of the plot
    if time_stamp is not None:
        ax.text(0.02, 0.97, time_stamp, ha='left', va='top',
                transform=ax.transAxes, fontsize=12, color=arrow_color)

    # If a calibration distance was specified, add units to the labels
    if calib_dist is not None and units == 'px':
        raise ValueError('Units must be specified if a calibration distance '
                         'is given.')
    ax.set_xlabel(f'{axis_labels[0]} ({units})')
    ax.set_ylabel(f'{axis_labels[1]} ({units})')

    # If an arrow scale was specified, add it to the title
    if arrow_scale != 1 and calib_dist is not None and title is not None:
        title = title + f' (arrows scaled ×{arrow_scale})'
    ax.set_title(title)
    # plt.show()


    return fig, ax


def plot_displacements(displacements,
                       highlight_radius_range=None,
                       highlight_angle_range=None,
                       highlight_color='b', calib_dist=None, units=None,
                       legend=None):
    """
    Plot the displacement vectors.

    This function takes in the same keyword arguments as plot_flow_field,
    but ignores some of them.

    PARAMETERS:
        displacements (np.array): Displacement vectors [j, i, y/x].
        highlight_radius_range (list): Range of magnitudes to highlight.
        highlight_angle_range (list): Range of angles to highlight.
        highlight_color (str): Color of the highlighted arrows and dots.
        calib_dist (float): Calibration distance.
        units (str): Units of the calibration distance.
        legend (list): Legend of the plot.

    RETURNS:
        fig (plt.figure): Figure object.
        ax (plt.axis): Axis object.
    """

    # Set all default values
    if legend is None:
        legend = ['Highlighted', 'Out of range']
    if highlight_angle_range is None:
        highlight_angle_range = [np.nan, np.nan]
    if highlight_radius_range is None:
        highlight_radius_range = [np.nan, np.nan]

    # Plot all displacement vectors
    fig, ax = plt.subplots()

    # If a calibration distance was specified, calibrate all values
    if calib_dist is not None:
        displacements = displacements * calib_dist

    # Get a list of indices that should be coloured
    highlight = filter_displacements(displacements,
                                     radius_range=highlight_radius_range,
                                     angle_range=highlight_angle_range)

    # Plot the indices below the threshold as dots
    ax.scatter(displacements[highlight, 1], displacements[highlight, 0],
               marker='^', s=10, color=highlight_color)
    ax.scatter(displacements[~highlight, 1], displacements[~highlight, 0],
               marker='o', s=10, color='k')

    # Draw zero lines
    ax.axhline(0, color='darkgrey', lw=0.5)
    ax.axvline(0, color='darkgrey', lw=0.5)

    # Pad the limits, but use only finite values
    displacements_finite = displacements[np.any(np.isfinite(displacements),
                                                axis=2)]
    ax.set_xlim([np.nanmin(displacements_finite[:, 1]) - 1,
                 np.nanmax(displacements_finite[:, 1]) + 1])
    ax.set_ylim([np.nanmin(displacements_finite[:, 0]) - 1,
                 np.nanmax(displacements_finite[:, 0]) + 1])

    # Pad the x limits to make the plot square
    ax.set_xlim([np.amin([ax.get_xlim()[0], ax.get_ylim()[0]]),
                 np.amax([ax.get_xlim()[1], ax.get_ylim()[1]])])

    ax.set_aspect('equal')

    # If a calibration distance was specified, add units to the labels
    if calib_dist is not None:
        ax.set_xlabel(f'Δx [{units}]')
        ax.set_ylabel(f'Δy [{units}]')
    else:
        ax.set_xlabel('Δx [px]')
        ax.set_ylabel('Δy [px]')

    # If points were highlighted, add a legend
    if np.any(highlight):
        ax.legend(legend)

    ax.set_title('All displacement vectors')
    plt.show()

    return fig, ax


def filter_displacements(displacements, radius_range=None,
                         angle_range=None, template=None):
    """
    Filter out displacement vectors based on their magnitude and angle.

    PARAMETERS:
        displacements (np.array): Displacement vectors [j, i, y/x].
        radius_range (list): Range of magnitudes to keep.
        angle_range (list): Range of angles to keep.
        template (np.array): Template to use for filtering.

    RETURNS:
        mask (np.array): Boolean mask [j, i] of the filtered vectors.
    """

    # MODE 1: Polar coordinates
    if template is None:
        # Set default values
        if angle_range is None:
            angle_range = [-np.pi, np.pi]
        if radius_range is None:
            radius_range = [0, np.inf]

        # Calculate the magnitude and angle of the displacement vectors
        magnitudes = np.linalg.norm(displacements, axis=2)
        angles = np.arctan2(displacements[:, :, 1], displacements[:, :, 0])

        # If only nans are given, skip the filtering
        if np.all(np.isnan(radius_range + angle_range)):
            mask = np.zeros(displacements.shape[:2], dtype=bool)

        # Filter the displacements based on the given radius and angle ranges
        else:
            # Create a mask the same size as displacements
            mask = np.ones(displacements.shape[:2], dtype=bool)

            if not np.isnan(radius_range[0]):
                mask = mask & (magnitudes > radius_range[0])
            if not np.isnan(radius_range[1]):
                mask = mask & (magnitudes < radius_range[1])
            if not np.isnan(angle_range[0]):
                mask = mask & (angles > angle_range[0])
            if not np.isnan(angle_range[1]):
                mask = mask & (angles < angle_range[1])

        # Return the mask
        return mask

    # MODE 2: Template-based filtering
    else:
        raise NotImplementedError('Template-based filtering is not implemented')


def optical_flow(images, slice_ct, window_ct, max_shift_px,
                 margins=[0, 0, 0, 0], valid_angles=[np.nan, np.nan],
                 background=None, sum_rows=False, use_guess=False, timing=False,
                 do_flow_plot=False, do_displ_plot=False, do_print_mean=False):
    """
    Perform particle image velocimetry (PIV) on a set of images.

    This function calculates the optical flow field between pairs of images
    using a three-pass PIV algorithm. The first pass calculates the
    displacement of the entire image, the second pass calculates the
    displacement of slices, and the third pass calculates the displacement of
    windows with subpixel accuracy. The displacements are then filtered based
    on their magnitude and angle.

    Idea: Maybe remove points that do not match their neighbours

    Parameters
    ----------
    images : np.array
        Images [c, y, x].
    slice_ct : int
        Number of slices to divide the image into.
    window_ct : tuple of int
        Number of windows in the y and x directions.
    max_shift_px : int
        Maximum displacement in pixels.
    margins : list of int, optional
        Number of pixels to cut off from each side of the image [top, bottom,
        left, right]. Default [0, 0, 0, 0].
    valid_angles : list of float, optional
        Range of angles to keep [min, max]. Default: [np.nan, np.nan].
    background : np.array, optional
        Background image to subtract from the images. Default: None.
    sum_rows : bool, optional
        Whether to sum the correlation windows along the rows. Default: False.
    do_flow_plot : bool, optional
        Whether to plot the flow field. Default: False.
    do_displ_plot : bool, optional
        Whether to plot the displacement vectors. Default: False.
    do_print_mean : bool, optional
        Whether to print the mean displacement for each image. Default: False.

    Raises
    ------
    ValueError
        If there are less than two images, the image width is not divisible by
        the number of windows, or the number of windows is not divisible by the
        number of slices.

    Returns
    -------
    displacements : np.array
        Displacement vectors [c, j, i, y/x].
    """

    # Suppress the empty slice warning
    warnings.filterwarnings(action='ignore', message='Mean of empty slice')

    # Check whether there are two or more images
    nr_images = images.shape[0]
    if nr_images < 2:
        # Error
        raise ValueError('At least two images are required for PIV.')

    # Check whether the width of the image is divisible by the number of windows
    if (images.shape[2] - margins[2] - margins[3]) % window_ct[1] != 0:
        raise ValueError(
                'Image width is not divisible by the number of windows')

    # Check whether the number of windows is divisible by the number of slices
    if window_ct[0] % slice_ct != 0:
        raise ValueError(
                'Number of windows is not divisible by the number of slices')

    # Subtract background
    if background is not None:
        images = subtract_background(images, background=background,
                                     timing=timing)

    # Cut off a number of pixels in each direction given by margins
    images_crop = images[:, margins[0]:(images.shape[1] - margins[1]),
                  margins[2]:(images.shape[2] - margins[3])]

    # In the cropped images, calculate the center of each window
    window_size = (images_crop[0].shape /
                   np.array([window_ct[0], window_ct[1]]))
    window_y = np.arange(window_size[0] / 2, images_crop[0].shape[0],
                         window_size[0])
    window_x = np.arange(window_size[1] / 2, images_crop[0].shape[1],
                         window_size[1]) \
        if not sum_rows else [images_crop[0].shape[1] / 2]
    window_centers = np.array([[[y, x] for x in window_x] for y in window_y])

    # Pre-allocate displacements array
    displacements = np.empty(
            (nr_images - 1, window_ct[0], window_ct[1], 2), dtype=np.float64) \
        if not sum_rows else np.empty(
            (nr_images - 1, window_ct[0], 1, 2), dtype=np.float64)

    # Loop over frames
    for c in trange(nr_images - 1, desc='Optical flow',
                    disable=not timing):

        # FIRST PASS
        # If there is a previous frame...
        # i -f use_guess and c > 0 and not np.all(np.isnan(displacements[c 1])):
        if use_guess and c > 0:
            # Use the mean horizontal displacement of the previous frame
            hor_disp_init = np.round(
                    np.nanmean(displacements[c - 1, :, :, 1]).flatten())

            # if np.isnan(hor_disp_init):
            #     print('nan')

            # If this is a nan, use a frame before this
            c_prev = 1
            while np.isnan(hor_disp_init) & (c > c_prev):
                hor_disp_init = np.round(
                        np.nanmean(
                                displacements[c - c_prev, :, :, 1]).flatten())
                c_prev += 1

            # Taking out cases where this gives 0, gives worse results

        # The first frame always uses the entire image to calculate
        # an initial displacement, also do this when the guess is unusable
        if c == 0 or (use_guess and np.isnan(hor_disp_init)) or not use_guess:
            # Calculate the displacement of the entire image
            corr_init = sig.correlate(images_crop[c + 1], images_crop[c])
            hor_disp_init = displacement_1d(corr_init, max_disp=max_shift_px)

        # If the correlation map has no peak, move to the next frame
        if np.isnan(hor_disp_init):
            continue

        # Otherwise, turn it into an integer
        hor_disp_init = int(hor_disp_init)

        # If there is movement, shift the second image to match the first
        images_shift = shift_displaced_image(images_crop[c:(c + 2)],
                                             hor_disp_init)

        # SECOND PASS
        # Divide the image into slices
        slice_set = np.array_split(images_shift, slice_ct, axis=1)
        # displacements = np.empty((window_ct[0], window_ct[1], 2),
        #                          dtype=np.float64) if not sum_rows \
        #     else np.empty((window_ct[0], 1, 2), dtype=np.float64)
        displacements[c, :, :, :] = np.nan

        # Get the horizontal displacement in the slices
        for j, slices in enumerate(slice_set):

            # Maximum displacement will be limited further,
            # with a minimum of 10 for peak detection purposes
            max_disp = hor_disp_init + 5 if hor_disp_init > 5 else 10
            # max_disp = max_disp

            # If the initial displacement was larger than 2 px, we will
            # prevent the next search from flipping back onto the original
            # peak at 0 movement
            ignore_only_if_mult = False if hor_disp_init > 2 else True

            # Calculate the displacement in the slice
            corr_slice = sig.correlate(slices[1], slices[0])
            hor_disp_slice = displacement_1d(
                    corr_slice, ignore_disp=-hor_disp_init,
                    ignore_only_if_mult=ignore_only_if_mult,
                    max_disp=max_disp, plot=False)

            # If this gives no result, there is no movement in this slice
            if np.isnan(hor_disp_slice):
                continue

            # Shift the second slice to match the first
            slices_shift = shift_displaced_image(slices, hor_disp_slice)

            # THIRD PASS
            # Divide the slice into windows
            window_set = np.array_split(slices_shift,
                                        window_ct[0] // slice_ct,
                                        axis=1)
            window_set = [np.array_split(row, window_ct[1], axis=2)
                          for row in window_set]

            # For each window, calculate the correlation maps
            corr_set = [[sig.correlate(window[1], window[0]) for window in row]
                        for row in window_set]

            if sum_rows:
                # Sum the correlation windows along the rows
                corr_set = [np.sum(row, axis=0, keepdims=True)
                            for row in corr_set]

            # Calculate the displacement from the 2d correlation maps
            disp_set = [[displacement_2d(corr, max_disp=max_disp)
                         for corr in row] for row in corr_set]

            # Put the displacements in the correct place in the array
            for j_w, row in enumerate(disp_set):
                for i, disp in enumerate(row):
                    displacements[c, j_w + j * window_ct[0] // slice_ct, i, :] \
                        = disp + [0, hor_disp_slice + hor_disp_init]

        # PLOTTING ACTION
        # Plot the displacement vectors
        if do_displ_plot:
            _, _ = plot_displacements(displacements[c, :, :, :],
                                      highlight_radius_range=[1e-20, np.inf],
                                      highlight_angle_range=valid_angles,
                                      legend=['Forward angles', 'Rejected'])

        # If the average horizontal displacement is larger than 1 px,
        # also replace displacements not in the valid angle range
        if np.abs(np.nanmean(displacements[c, :, :, 1])) > 1:
            valid_mask = filter_displacements(displacements[c, :, :, :],
                                              angle_range=valid_angles,
                                              radius_range=[1, np.inf])
            displacements[c, ~valid_mask] = np.nan

        # Plot flow field
        if do_flow_plot:
            _, _ = plot_flow_field(displacements[c, :, :, :], window_centers,
                                   arrow_scale=0.5,
                                   arrow_color='white',
                                   background=images_crop[0])

        # Print mean horizontal velocity
        if do_print_mean and not np.all(np.isnan(displacements[c, :, :, 1])):
            print(np.nanmean(displacements[c, :, :, 1]))

    return displacements, window_centers


def batch_optical_flow(position_nr, series_nrs, slice_ct, window_ct,
                       max_shift_px, margins, file_range=None,
                       valid_angles=[np.pi / 4, 3 * np.pi / 4],
                       sum_rows=True, use_guess=True, timing=True,
                       do_mean_plot=True):
    """
    Perform optical flow on a batch of measurement series at a given position.

    Parameters
    ----------
    position_nr : int
        Position number.
    series_nrs : list of int
        Series numbers.
    slice_ct : int
        Number of slices to divide the image into.
    window_ct : tuple of int
        Number of windows in the y and x directions.
    max_shift_px : int
        Maximum displacement in pixels.
    margins : list of int
        Number of pixels to cut off from each side of the image [top, bottom,
        left, right].
    file_range : int | tuple
        Number of images to read or a tuple of start and stop indices.
        Default: None.
    valid_angles : list of float, optional
        Range of angles to keep [min, max]. Default: [np.pi / 4, 3 * np.pi / 4].
    sum_rows : bool, optional
        Whether to sum the correlation windows along the rows. Default: True.
    use_guess : bool, optional
        Whether to use the mean horizontal displacement of the previous frame.
        Default: True.
    timing : bool, optional
        Whether to show a progress bar. Default: True.
    do_mean_plot : bool, optional
        Whether to plot the mean displacement of each series. Default: True.

    Returns
    -------
    None
    """

    for series_nr in series_nrs:

        # Announce start
        print(f'Processing pos{position_nr}-{series_nr}')
        print('-----------------')

        # Load background
        background = cv.imread(f'data/backgrounds/pos{position_nr}'
                               f'-{series_nr}.tif', cv.IMREAD_GRAYSCALE)

        # Load images
        images = read_image_directory(f'data/pos{position_nr}-{series_nr}',
                                      image_type='tif', timing=True,
                                      file_range=file_range)

        # Perform optical flow
        displ, window_cent = optical_flow(
                images, slice_ct, window_ct, max_shift_px, margins,
                valid_angles=valid_angles, background=background,
                sum_rows=sum_rows, use_guess=use_guess, timing=timing)

        # Calculate mean displacement
        displ_mean = mean_displacement(displ, comp='x', av_dir='all')

        # Save the displacement, mean displacement, and window centers
        np.save(f'processed/pos{position_nr}-{series_nr}_displ.npy',
                displ)
        np.save(f'processed/pos{position_nr}-{series_nr}_displ_av.npy',
                displ_mean)
        np.save(f'processed/pos{position_nr}-{series_nr}_window_pos.npy',
                window_cent)

        # Plot the mean displacement
        if do_mean_plot:
            fig, ax = plt.subplots()
            ax.plot(displ_mean.flatten())
            ax.set_xlabel('Frame')
            ax.set_ylabel('Mean displacement [px]')
            ax.set_ylim([-10, 50])
            plt.show()

        # Delete the images and displacements to save memory
        del images, background, displ, displ_mean, window_cent


def mean_displacement(displacements, comp='norm', av_dir='ij'):
    """
    Calculate the mean displacement of the flow field.

    To calculate horizontal flow profiles, set comp='x' and av_dir='i'.

    Parameters
    ----------
    displacements : np.array
        Displacement vectors [c, j, i, y/x].
    comp : str, optional
        Component of the displacement to calculate. Default: 'norm'.
    av_dir : str, optional
        Direction in which to calculate the mean. Default: 'all'.

    Returns
    -------

    """

    # If displacements is 3D, add a dimension
    if displacements.ndim == 3:
        displacements = displacements[np.newaxis, :, :, :]

    # Check whether the component and direction are valid
    if comp not in ['norm', 'x', 'y', 'xy']:
        raise ValueError('Invalid component designation')
    if av_dir not in ['all', 'i', 'j']:
        raise ValueError('Invalid direction designation')

    # Set dimensions of the output array
    xy_length = 1 if comp in ['norm', 'x', 'y'] else 2
    j_length = displacements.shape[1] if av_dir == 'i' else 1
    i_length = displacements.shape[2] if av_dir == 'j' else 1

    # Pre-allocate array of mean displacements
    mean_displacements = np.empty((displacements.shape[0],
                                   j_length, i_length, xy_length))

    # Loop through frames
    for c in range(len(displacements)):
        # If all entries are nan, the mean displacement is nan
        if np.all(np.isnan(displacements[c, :, :, :])):
            mean_displacements[c, :, :, :] = np.nan
            continue

        # If the magnitude is requested, first calculate the norm
        if comp == 'norm':
            entries = np.linalg.norm(displacements[c, :, :, :], axis=2)
        elif comp == 'x':
            entries = displacements[c, :, :, 1]
        elif comp == 'y':
            entries = displacements[c, :, :, 0]
        elif comp == 'xy':
            entries = displacements[c, :, :, :]

        # Check in which direction the mean should be calculated
        if av_dir == 'all':

            # If we want a scalar output, take the mean, otherwise
            # separately in the x and y directions
            if comp in ['norm', 'x', 'y']:
                mean_displacements[c, 0, 0, :] = np.nanmean(entries)
            else:
                entries = displacements[c, :, :, :]
                mean_displacements[c, 0, 0, :] = (
                    np.nanmean(entries, axis=(0, 1)))

        elif av_dir == 'j':

            for i in range(i_length):
                # If we want a scalar output, take the mean, otherwise
                # separately in the x and y directions
                if comp in ['norm', 'x', 'y']:
                    mean_displacements[c, 0, i, :] = np.nanmean(entries[:, i])
                else:
                    entries = displacements[c, :, i, :]
                    mean_displacements[c, 0, i, :] = (
                        np.nanmean(entries, axis=0))

        elif av_dir == 'i':

            for j in range(j_length):
                # If we want a scalar output, take the mean, otherwise
                # separately in the x and y directions
                if comp in ['norm', 'x', 'y']:
                    mean_displacements[c, j, 0, :] = np.nanmean(entries[j, :])
                else:
                    entries = displacements[c, j, :, :]
                    mean_displacements[c, j, 0, :] = (
                        np.nanmean(entries, axis=0))

    return mean_displacements
