"""Open one image, threshold its intensities, and compare histograms.

Edit the intensity thresholds near the top of the file, then run this script.
It opens a file picker, loads the selected image, zeroes out pixels outside the
allowed intensity range, and shows/saves a before/after comparison.
"""

from __future__ import annotations

from pathlib import Path

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from tcm_utils.file_dialogs import ask_open_file
from tcm_utils.io_utils import load_image


# Intensity filter settings.
LOWER_INTENSITY = 2
UPPER_INTENSITY = 64

# Small blur applied before thresholding.
BLUR_SIGMA = 32

# Plot settings.
FIGSIZE = (15, 9)
IMAGE_CMAP = "gray"
IMAGE_VMIN = None
IMAGE_VMAX = UPPER_INTENSITY
ZOOM_X_MIN = 400
ZOOM_X_MAX = 600
ZOOM_Y_MIN = 600
ZOOM_Y_MAX = 400
COUNT_X_MIN = None
COUNT_X_MAX = 50
COUNT_Y_MIN = 0
COUNT_Y_MAX = None
FILTERED_COUNT_Y_MAX = None
SCATTER_MARKER = "."
SCATTER_SIZE = 6
SCATTER_ALPHA = 0.8
COUNT_LINESTYLE = "-"
COUNT_LINEWIDTH = 1.0
ORIGINAL_SCATTER_COLOR = "tab:blue"
FILTERED_SCATTER_COLOR = "tab:green"
THRESHOLD_COLOR = "tab:red"
THRESHOLD_LINESTYLE = "--"
THRESHOLD_LINEWIDTH = 1.5
TITLE = "Intensity threshold filter"
ORIGINAL_TITLE = "Original"
BLURRED_TITLE = "After background subtraction"
FILTERED_TITLE = "After filter"
BEFORE_HIST_TITLE = "Counts before background subtraction"
BLURRED_HIST_TITLE = "Counts after background subtraction"
AFTER_HIST_TITLE = "Counts after filter"
IMAGE_ALPHA = 1.0
GRID_ALPHA = 0.25

# Optional output settings.
SAVE_FILTERED_IMAGE = False
SAVE_FIGURE = False
FIGURE_DPI = 180


def apply_intensity_filter(image: np.ndarray, lower: float, upper: float) -> np.ndarray:
    """Return a copy of ``image`` clipped to ``[lower, upper]``.

    Values below ``lower`` are raised to ``lower``.
    """

    if lower > upper:
        raise ValueError("LOWER_INTENSITY must be <= UPPER_INTENSITY")

    arr = np.asarray(image)
    filtered = np.clip(arr, lower, upper)
    return filtered.astype(arr.dtype, copy=False)


def blur_image(image: np.ndarray, sigma: int) -> np.ndarray:
    """Apply a small Gaussian blur before thresholding."""

    if sigma < 1 == 0:
        raise ValueError("BLUR_SIGMA must be a positive integer")

    return cv.GaussianBlur(np.asarray(image), (0, 0), sigmaX=sigma)


def scatter_counts(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return x/y values for a scatter plot of exact intensity counts."""

    arr = np.asarray(image).ravel()
    if arr.size == 0:
        return np.asarray([]), np.asarray([])

    if np.issubdtype(arr.dtype, np.integer):
        offset = int(arr.min())
        counts = np.bincount(arr.astype(np.int64) - offset)
        x_values = np.arange(offset, offset + counts.size)
        return x_values, counts

    x_values, counts = np.unique(arr, return_counts=True)
    return x_values, counts


def apply_zoom(
    ax: Axes,
    x_min: float | None,
    x_max: float | None,
    y_min: float | None,
    y_max: float | None,
) -> None:
    """Apply a shared zoom region to an image axis when bounds are provided."""

    if x_min is not None and x_max is not None:
        ax.set_xlim(x_min, x_max)
    if y_min is not None and y_max is not None:
        ax.set_ylim(y_max, y_min)


def plot_comparison(
    original: np.ndarray,
    blurred: np.ndarray,
    filtered: np.ndarray,
    lower: float,
    upper: float,
    output_path: Path,
) -> None:
    """Plot the original, blurred, and filtered images with exact-count scatter plots."""

    fig, axes = plt.subplots(2, 3, figsize=FIGSIZE, constrained_layout=True)
    x_original, y_original = scatter_counts(original)
    x_blurred, y_blurred = scatter_counts(blurred)
    x_filtered, y_filtered = scatter_counts(filtered)

    x_min = COUNT_X_MIN
    x_max = COUNT_X_MAX
    if x_min is None:
        x_min = float(np.nanmin(original))
    if x_max is None:
        x_max = float(np.nanmax(original))

    y_max_shared = COUNT_Y_MAX
    if y_max_shared is None:
        y_max_shared = float(np.nanmax(y_original) if y_original.size else 0)
        y_max_shared = 1.0 if y_max_shared <= 0 else y_max_shared * 1.05

    y_max_filtered = FILTERED_COUNT_Y_MAX
    if y_max_filtered is None:
        filtered_nonlower = y_filtered[x_filtered != lower]
        y_max_filtered = float(
            np.nanmax(filtered_nonlower) if filtered_nonlower.size else 0)
        y_max_filtered = 1.0 if y_max_filtered <= 0 else y_max_filtered * 1.05

    axes[0, 0].imshow(
        original,
        cmap=IMAGE_CMAP,
        vmin=IMAGE_VMIN,
        vmax=IMAGE_VMAX,
        alpha=IMAGE_ALPHA,
    )
    axes[0, 0].set_title(ORIGINAL_TITLE)
    axes[0, 0].set_axis_off()
    apply_zoom(axes[0, 0], ZOOM_X_MIN, ZOOM_X_MAX, ZOOM_Y_MIN, ZOOM_Y_MAX)

    axes[0, 1].imshow(
        blurred,
        cmap=IMAGE_CMAP,
        vmin=IMAGE_VMIN,
        vmax=IMAGE_VMAX,
        alpha=IMAGE_ALPHA,
    )
    axes[0, 1].set_title(BLURRED_TITLE)
    axes[0, 1].set_axis_off()
    apply_zoom(axes[0, 1], ZOOM_X_MIN, ZOOM_X_MAX, ZOOM_Y_MIN, ZOOM_Y_MAX)

    axes[0, 2].imshow(
        filtered,
        cmap=IMAGE_CMAP,
        vmin=IMAGE_VMIN,
        vmax=IMAGE_VMAX,
        alpha=IMAGE_ALPHA,
    )
    axes[0, 2].set_title(f"{FILTERED_TITLE}\n[{lower:g}, {upper:g}] kept")
    axes[0, 2].set_axis_off()
    apply_zoom(axes[0, 2], ZOOM_X_MIN, ZOOM_X_MAX, ZOOM_Y_MIN, ZOOM_Y_MAX)

    axes[1, 0].plot(
        x_original,
        y_original,
        color=ORIGINAL_SCATTER_COLOR,
        linestyle=COUNT_LINESTYLE,
        linewidth=COUNT_LINEWIDTH,
        marker=SCATTER_MARKER,
        alpha=SCATTER_ALPHA,
    )
    axes[1, 0].axvline(lower, color=THRESHOLD_COLOR,
                       linestyle=THRESHOLD_LINESTYLE, linewidth=THRESHOLD_LINEWIDTH)
    axes[1, 0].axvline(upper, color=THRESHOLD_COLOR,
                       linestyle=THRESHOLD_LINESTYLE, linewidth=THRESHOLD_LINEWIDTH)
    axes[1, 0].set_title(BEFORE_HIST_TITLE)
    axes[1, 0].set_xlabel("Intensity")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].set_xlim(x_min, x_max)
    axes[1, 0].set_ylim(COUNT_Y_MIN, y_max_shared)
    axes[1, 0].grid(alpha=GRID_ALPHA)

    axes[1, 1].plot(
        x_blurred,
        y_blurred,
        color="tab:orange",
        linestyle=COUNT_LINESTYLE,
        linewidth=COUNT_LINEWIDTH,
        marker=SCATTER_MARKER,
        alpha=SCATTER_ALPHA,
    )
    axes[1, 1].axvline(lower, color=THRESHOLD_COLOR,
                       linestyle=THRESHOLD_LINESTYLE, linewidth=THRESHOLD_LINEWIDTH)
    axes[1, 1].axvline(upper, color=THRESHOLD_COLOR,
                       linestyle=THRESHOLD_LINESTYLE, linewidth=THRESHOLD_LINEWIDTH)
    axes[1, 1].set_title(BLURRED_HIST_TITLE)
    axes[1, 1].set_xlabel("Intensity")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].set_xlim(x_min, x_max)
    axes[1, 1].set_ylim(COUNT_Y_MIN, y_max_shared)
    axes[1, 1].grid(alpha=GRID_ALPHA)

    axes[1, 2].plot(
        x_filtered,
        y_filtered,
        color=FILTERED_SCATTER_COLOR,
        linestyle=COUNT_LINESTYLE,
        linewidth=COUNT_LINEWIDTH,
        marker=SCATTER_MARKER,
        alpha=SCATTER_ALPHA,
    )
    axes[1, 2].axvline(lower, color=THRESHOLD_COLOR,
                       linestyle=THRESHOLD_LINESTYLE, linewidth=THRESHOLD_LINEWIDTH)
    axes[1, 2].axvline(upper, color=THRESHOLD_COLOR,
                       linestyle=THRESHOLD_LINESTYLE, linewidth=THRESHOLD_LINEWIDTH)
    axes[1, 2].set_title(AFTER_HIST_TITLE)
    axes[1, 2].set_xlabel("Intensity")
    axes[1, 2].set_ylabel("Count")
    axes[1, 2].set_xlim(x_min, x_max)
    axes[1, 2].set_ylim(COUNT_Y_MIN, y_max_shared)
    axes[1, 2].grid(alpha=GRID_ALPHA)

    fig.suptitle(TITLE, fontsize=14)

    if SAVE_FIGURE:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=FIGURE_DPI)

    plt.show()


def main() -> None:
    # image_path = ask_open_file(
    #     key="intensity_filter_image",
    #     title="Select an image to filter",
    #     filetypes=(
    #         ("Image files", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp"),
    #         ("All files", "*.*"),
    #     ),
    # )
    image_path = Path(
        "/Volumes/Data/PIV/260820_piv/step_1-5bar/260831_152310_step_1-5bar_20-0mA/P-001/P-001_20260831_165049004160.tif")
    if image_path is None:
        print("No image selected.")
        return

    image = load_image(image_path)
    background = blur_image(image, BLUR_SIGMA)

    # Subtract the blurred image from the original to remove background illumination.
    subtracted = np.subtract(image, background, dtype=np.int32)
    subtracted = np.clip(subtracted, 0, None).astype(image.dtype, copy=False)

    filtered = apply_intensity_filter(
        subtracted, LOWER_INTENSITY, UPPER_INTENSITY)

    image_path = Path(image_path)
    print(f"Loaded {image_path}")
    print(f"  dtype: {image.dtype}")
    print(f"  original range: {np.nanmin(image):g} to {np.nanmax(image):g}")
    print(
        f"  background range: {np.nanmin(background):g} to {np.nanmax(background):g}")
    print(
        f"  bg subtracted range: {np.nanmin(subtracted):g} to {np.nanmax(subtracted):g}")
    print(f"  blur kernel: {BLUR_SIGMA}x{BLUR_SIGMA}")
    print(f"  threshold range: {LOWER_INTENSITY:g} to {UPPER_INTENSITY:g}")
    print(
        f"  filtered range: {np.nanmin(filtered):g} to {np.nanmax(filtered):g}")

    if SAVE_FILTERED_IMAGE:
        filtered_path = image_path.with_name(
            f"{image_path.stem}_thresholded{image_path.suffix}")
        if filtered_path != image_path:
            from tifffile import imwrite

            imwrite(filtered_path, filtered)
            print(f"Saved filtered image to {filtered_path}")

    output_figure = image_path.with_name(
        f"{image_path.stem}_intensity_histogram.png")
    plot_comparison(image, subtracted, filtered, LOWER_INTENSITY,
                    UPPER_INTENSITY, output_figure)
    if SAVE_FIGURE:
        print(f"Saved figure to {output_figure}")


if __name__ == "__main__":
    main()
