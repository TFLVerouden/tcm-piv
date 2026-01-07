import numpy as np
import matplotlib.pyplot as plt
import piv_functions as piv

N=20
# Randomly generate a 4D array of shape (N, N, N, 2)
coords = np.random.rand(N, N, N, 2) * 100  # Scale to 100 for visibility

# Apply the filter_neighbours function
coords_filt = piv.filter_neighbours(coords, thr=0.1, n_nbs=(2, 2, 2))

# Count the number of NaN values in the filtered coordinates
n_nan = np.sum(np.isnan(coords_filt))
print(f"Number of NaN values in filtered coordinates: {n_nan}")

# Plot a single frame (frame 5) to visualize valid and invalid coordinates
frame_idx = N//2  # Middle frame
frame_coords = coords_filt[frame_idx, :, :, :]

# Create a mask for valid coordinates (not NaN)
valid_mask = ~np.isnan(frame_coords[:, :, 0])

# Get valid and invalid coordinates
valid_coords = frame_coords[valid_mask]
invalid_coords = frame_coords[~valid_mask]

# Create the plot
fig, ax = plt.subplots(figsize=(10, 8))

# Plot valid coordinates in blue
if len(valid_coords) > 0:
    ax.scatter(valid_coords[:, 1], valid_coords[:, 0], 
              c='blue', s=30, alpha=0.7, label=f'Valid ({len(valid_coords)})')

# Plot invalid coordinates in red (just the positions where they would be)
invalid_y, invalid_x = np.where(~valid_mask)
if len(invalid_y) > 0:
    ax.scatter(invalid_x, invalid_y, 
              c='red', s=30, alpha=0.7, marker='x', 
              label=f'Invalid ({len(invalid_y)})')

ax.set_xlabel('X coordinate')
ax.set_ylabel('Y coordinate')
ax.set_title(f'Coordinate filtering results for frame {frame_idx}')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"Frame {frame_idx}: {len(valid_coords)} valid, {len(invalid_y)} invalid coordinates")