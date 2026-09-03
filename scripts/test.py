from tcm_piv.preprocessing import denoise
from tcm_utils.io_utils import load_images
from pathlib import Path
from matplotlib import pyplot as plt

img1_nr = "0001"
img2_nr = "4160"
img1 = Path(
    "/Volumes/Data/PIV/260820_piv/step_1-5bar/260831_152310_step_1-5bar_20-0mA/P-001/P-001_20260831_16504900" + img1_nr + ".tif")
img2 = Path("/Volumes/Data/PIV/260820_piv/step_1-5bar/260831_152310_step_1-5bar_20-0mA/P-001/P-001_20260831_16504900" + img2_nr + ".tif")
images = load_images([img1, img2])

denoised_images = denoise(images, lower_intensity=2, upper_intensity=4095,
                          shift_to_zero=True, reduce_dtype=True, debug=True)

# Show a comparison on a roi of both images
x_min, x_max = 400, 600
y_min, y_max = 400, 600

v_min, v_max = 0, 128

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].imshow(images[0][y_min:y_max, x_min:x_max],
                 cmap='gray', vmin=v_min, vmax=v_max)
axs[0, 0].set_title('Original Image 1')
axs[0, 1].imshow(images[1][y_min:y_max, x_min:x_max],
                 cmap='gray', vmin=v_min, vmax=v_max)
axs[0, 1].set_title('Original Image 2')
axs[1, 0].imshow(denoised_images[0][y_min:y_max, x_min:x_max],
                 cmap='gray', vmin=v_min, vmax=v_max)
axs[1, 0].set_title('Denoised Image 1')
axs[1, 1].imshow(denoised_images[1][y_min:y_max, x_min:x_max],
                 cmap='gray', vmin=v_min, vmax=v_max)
axs[1, 1].set_title('Denoised Image 2')

fig.tight_layout()

if plt.get_backend().lower() == 'agg':
    output_path = Path(__file__).with_name('test_roi_comparison.png')
    fig.savefig(output_path, dpi=200)
    print(f'Saved comparison figure to {output_path}')
else:
    plt.show()
