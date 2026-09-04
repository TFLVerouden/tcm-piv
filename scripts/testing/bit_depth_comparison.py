import os
from tcm_utils.cvd_check import set_cvd_friendly_colors, get_color
import numpy as np
import matplotlib.pyplot as plt
import tifffile

set_cvd_friendly_colors()

# 1. Define folder path and filenames
folder_path = os.path.join("scripts", "bit_depth_samples")
files = {
    "8-bit": "8bit000001.tif",
    "12-bit Real": "12bit_real000001.tif",
    "16-bit Lower": "16bit_lower000001.tif",
    "16-bit Higher": "16bit_higher000001.tif"
}

# 2. Load images
images = {}
for name, filename in files.items():
    path = os.path.join(folder_path, filename)
    images[name] = tifffile.imread(path)

# 3. Create a 2x2 plotting grid
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
axes = axes.ravel()

# 4. Loop through each image to compute data and plot histograms
for i, (name, img) in enumerate(images.items()):
    # Get metadata
    dtype = img.dtype
    unique_values = np.unique(img)
    num_unique = len(unique_values)

    print(f"\n[{name}]")
    print(f"  Data type: {dtype}")
    print(f"  Unique gray levels found: {num_unique}")
    if num_unique > 0:
        print(f"  Min value: {img.min()} | Max value: {img.max()}")

    # Determine optimal histogram range based on format
    if name == "8-bit":
        bins = 256
        hist_range = (0, 256)
        zoom_limit = 2  # Zoom close to 0 since it's a dim image
        color = get_color(0)
    elif name == "12-bit Real":  # 12-bit Real and 16-bit Lower occupy the 0-4095 range
        bins = 4096
        hist_range = (0, 4096)
        zoom_limit = 32  # Zoom into the very dim noise floor/peaks
        color = get_color(2)
    elif name == "16-bit Higher":
        bins = 65536
        hist_range = (0, 65536)
        zoom_limit = 512
        color = get_color(1)
    elif name == "16-bit Lower":
        bins = 65536
        hist_range = (0, 65536)
        zoom_limit = 512  # Zoom into the very dim noise floor/peaks
        color = get_color(3)

    # Calculate and plot histogram
    hist, bins_edges = np.histogram(img, bins=bins, range=hist_range)
    axes[i].scatter(bins_edges[:-1], hist, s=5, color=color)

    # Visual formatting
    axes[i].set_title(f"{name} Histogram ({dtype})",
                      fontsize=12, fontweight='bold')
    axes[i].set_xlabel("Intensity")
    axes[i].set_ylabel("Count")
    # Zoomed x-axis to clearly inspect dim peaks
    axes[i].set_xlim(-zoom_limit/8, zoom_limit)
    axes[i].grid(True, alpha=0.3)
    # Change y axis to use scientific notation for better readability
    axes[i].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

    # Only display y axis labels on leftmost plots and x axis labels on bottom plots
    if i % 2 != 0:
        axes[i].set_ylabel("")
    if i < 2:
        axes[i].set_xlabel("")

plt.tight_layout()
plt.savefig(os.path.join(folder_path, "histogram_comparison.png"), dpi=300)
plt.show()
