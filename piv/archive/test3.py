import numpy as np

# Generate a random array of shape (2, 10, 10, 10)
random_array = np.random.rand(2, 10, 10, 10)

# Calculate the median and standard deviation along the last axis
med = np.nanmedian(random_array, axis=(1, 2, 3))
std = np.nanstd(random_array, axis=(1, 2, 3))

print("Median:", med)
print("Standard Deviation:", std)

# Calculate vector length, get median and standard deviation
vec_length = np.sqrt(np.sum(random_array**2, axis=-1))
print("Vector Length:", vec_length)
vec_length_med = np.nanmedian(vec_length, axis=(1, 2))
vec_length_std = np.nanstd(vec_length, axis=(1, 2))
print("Vector Length Median:", vec_length_med)
print("Vector Length Standard Deviation:", vec_length_std)

# Length of median
vec_length_med_len = np.sqrt(np.sum(med**2))
print("Length of Median Vector:", vec_length_med_len)