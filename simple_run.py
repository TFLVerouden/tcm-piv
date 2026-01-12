import matplotlib.pyplot as plt
import numpy as np
import tcm_piv.init_config as cfg
import tcm_piv as piv
from pathlib import Path
from tcm_utils.io_utils import load_images
import scipy.signal as sig

cfg_path = Path("examples/image_pair/input/config_image_pair.toml")
cfg_path = cfg_path.resolve()

cfg.read_file(cfg_path)

# 1. LOAD IMAGES
imgs = load_images(cfg.IMAGE_LIST)

# Plot the two images normalised to the total min/max over both images


def _imshow_norm(ax, img):
    vmin = float(np.min(img))
    vmax = float(np.max(img))
    ax.imshow(img, cmap="gray", vmin=vmin, vmax=vmax)


fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
_imshow_norm(axes[0], imgs[0])
axes[0].set_title("Image 0")
_imshow_norm(axes[1], imgs[1])
axes[1].set_title("Image 1")
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

# 2. CALCULATE CORRELATION MAP
corrs = piv.calc_corr(0, imgs, shifts=np.array(
    [[0, 0], [0, 0]]), n_wins=(1, 1), overlap=0)

# Show correlation map, again normalised
fig2, ax2 = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)
corr_map, center = corrs[(0, 0, 0)]
vmax = float(np.nanmax(corr_map))
ax2.imshow(corr_map, cmap="magma", vmin=0, vmax=vmax)
ax2.set_title("Correlation map")
ax2.plot(center[1], center[0], "c+", markersize=10, markeredgewidth=2)
ax2.set_xticks([])
ax2.set_yticks([])
plt.show()

# 3. FIND PEAKS
peaks, _ = piv.find_disps(corrs, n_peaks=cfg.NR_PEAKS[0], subpx=True)
print("Detected peaks (y, x):", peaks)

# # --- Load image pair ---
# image_files = _list_tifs(cfg.image_dir)
# if len(image_files) < 2:
#     raise RuntimeError(f"Need at least 2 images in {cfg.image_dir}")

# img0 = _load_grayscale(image_files[0])
# img1 = _load_grayscale(image_files[1])
# imgs = np.stack([img0, img1], axis=0)

# # --- Calibration ---
# timestep_s, scale_m_per_px = _load_camera_calibration(
#     camera_json=cfg.camera_metadata_json,
#     calib_json=cfg.calib_metadata_json,
# )

# # --- Preprocess (crop + background subtraction) ---
# # Keep a crop-only version for visual sanity checks.
# imgs_crop, _ = _apply_crop_and_background(
#     imgs,
#     crop_roi=cfg.crop_roi,
#     background_path=None,
# )

# imgs_pp, bg = _apply_crop_and_background(
#     imgs,
#     crop_roi=cfg.crop_roi,
#     background_path=cfg.background_path,
# )

# # --- Visualise input + preprocessing ---
# fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
# _imshow_norm(axes[0, 0], img0)
# axes[0, 0].set_title("Raw img0")
# _imshow_norm(axes[0, 1], img1)
# axes[0, 1].set_title("Raw img1")

# _imshow_norm(axes[1, 0], imgs_pp[0])
# axes[1, 0].set_title("Preprocessed img0")

# if bg is not None:
#     _imshow_norm(axes[1, 1], bg)
#     axes[1, 1].set_title("Background")
# else:
#     axes[1, 1].axis("off")

# for ax in axes.ravel():
#     ax.set_xticks([])
#     ax.set_yticks([])

# # --- Window geometry preview (same splitting used in correlation) ---
# # Use the *same* image geometry as the correlation step (downsampled if requested).
# # For plotting only: if background subtraction wipes the image to all-zeros,
# # fall back to the crop-only image so split_n_shift's preview doesn't divide by 0.
# img_for_preview = imgs_pp[0]
# if float(np.max(img_for_preview)) <= 0:
#     img_for_preview = imgs_crop[0]

# if cfg.downsample_factor > 1:
#     img_preview = piv.downsample(
#         img_for_preview[np.newaxis, ...], cfg.downsample_factor)[0]
#     win_pos_scale = float(cfg.downsample_factor)
# else:
#     img_preview = img_for_preview
#     win_pos_scale = 1.0

# piv.split_n_shift(
#     img_preview,
#     n_wins=cfg.nr_windows,
#     overlap=cfg.window_overlap,
#     shift=(0, 0),
#     shift_mode="before",
#     plot=True,
# )

# # --- Single pass correlation / peak detection ---
# corrs = piv.calc_corrs(
#     imgs_pp,
#     n_wins=cfg.nr_windows,
#     shifts=None,
#     overlap=cfg.window_overlap,
#     ds_fac=cfg.downsample_factor,
# )

# corrs_sum = piv.sum_corrs(
#     corrs,
#     cfg.corrs_to_sum,
#     n_wins=cfg.nr_windows,
#     shifts=None,
# )

# disp_unf, int_unf = piv.find_disps(
#     corrs_sum,
#     n_wins=cfg.nr_windows,
#     shifts=None,
#     n_peaks=cfg.nr_peaks,
#     ds_fac=cfg.downsample_factor,
#     min_dist=cfg.min_peak_distance,
#     verbose=True,
# )

# # --- Plot correlation maps + peaks (frame 0) ---
# wy, wx = cfg.nr_windows
# fig2, axes2 = plt.subplots(wy, wx, figsize=(
#     4 * wx, 4 * wy), constrained_layout=True)
# axes2 = np.atleast_2d(axes2)

# for j in range(wy):
#     for k in range(wx):
#         corr_map, center = corrs_sum[(0, j, k)]
#         ax = axes2[j, k]

#         # Display log-scaled to make peaks visible
#         corr_disp = np.log1p(np.maximum(corr_map.astype(np.float64), 0))
#         vmax = float(np.nanmax(corr_disp)) if corr_disp.size else 1.0
#         if not np.isfinite(vmax) or vmax <= 0:
#             vmax = 1.0
#         ax.imshow(corr_disp, cmap="magma", vmin=0, vmax=vmax)
#         ax.set_title(f"Corr (win {j},{k})")

#         # Mark correlation-map center
#         ax.plot(center[1], center[0], "c+", markersize=10, markeredgewidth=2)

#         # Re-find peaks just for plotting (so you can see what peak finder did)
#         peaks, _ = piv.find_peaks(
#             corr_map,
#             n_peaks=cfg.nr_peaks,
#             min_dist=cfg.min_peak_distance,
#         )
#         for p in range(cfg.nr_peaks):
#             if not np.isnan(peaks[p]).any():
#                 ax.plot(peaks[p, 1], peaks[p, 0], "wo", markersize=5)

#         ax.set_xticks([])
#         ax.set_yticks([])

# # --- Postprocessing (mirror old piv.py pass-1 style) ---
# vx_max, vy_max = cfg.max_velocity_vx_vy_m_s
# a_y = float(vy_max) * timestep_s / scale_m_per_px
# b_x = float(vx_max) * timestep_s / scale_m_per_px

# disp_glo_5d = np.asarray(piv.filter_outliers(
#     "semicircle_rect",
#     disp_unf,
#     a=a_y,
#     b=b_x,
#     verbose=True,
# ))
# disp_glo = piv.strip_peaks(disp_glo_5d, axis=-2, mode="reduce", verbose=True)

# n_corrs = int(disp_glo.shape[0])
# n_nbs = _clamp_n_nbs(cfg.neighbourhood_size, shape_3=(n_corrs, wy, wx))

# disp_final = piv.filter_neighbours(
#     disp_glo,
#     n_nbs=n_nbs,
#     thr=cfg.neighbourhood_threshold,
#     mode="xy",
#     replace=False,
#     verbose=True,
#     timing=False,
# )

# # --- Displacement/velocity field plot ---
# _, win_pos = piv.split_n_shift(
#     img_preview, cfg.nr_windows, overlap=cfg.window_overlap)
# # back to original pixel coordinates if downsampled
# win_pos = win_pos * win_pos_scale

# disp0 = disp_final[0]  # (wy, wx, 2) for the single pair
# vel0_m_s = disp0 * scale_m_per_px / timestep_s

# fig3, ax3 = plt.subplots(1, 1, figsize=(8, 6), constrained_layout=True)
# _imshow_norm(ax3, imgs_pp[0])
# ax3.set_title("Displacement vectors (single pair)")

# # Quiver expects (x, y) positions and (u, v) vectors.
# X = win_pos[:, :, 1]
# Y = win_pos[:, :, 0]
# U = disp0[:, :, 1]
# V = disp0[:, :, 0]

# ax3.quiver(X, Y, U, V, color="y", angles="xy", scale_units="xy", scale=1)

# # Add a small text box with one representative velocity number
# vmag = np.linalg.norm(vel0_m_s.reshape(-1, 2), axis=1)
# vmed = float(np.nanmedian(vmag)) if np.any(~np.isnan(vmag)) else float("nan")
# ax3.text(
#     0.01,
#     0.99,
#     f"dt = {timestep_s:.6g} s\nscale = {scale_m_per_px:.6g} m/px\n|v| median = {vmed:.3g} m/s",
#     transform=ax3.transAxes,
#     va="top",
#     ha="left",
#     color="w",
#     bbox={"facecolor": "black", "alpha": 0.5, "pad": 6},
# )

# ax3.set_xticks([])
# ax3.set_yticks([])

# print("Done (simple single-pass). Showing figures...")
# plt.show()
# return 0


# if __name__ == "__main__":
# raise SystemExit(main(sys.argv))
