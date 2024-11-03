import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from feature_extraction_methods import dummy_roi_extractor

# Extract the bounding box cancer region
bbox_cancer_region = dummy_roi_extractor.extract_cancer_roi()


# Gabor filter function
def apply_gabor_filter(image, k_size, sigma, theta, lambd, gamma, psi):
    """Use gabor filter to extract features from the image.

    Args:
    k_size : Kernel size
    sigma: is the standard deviation of the Gaussian function used in the Gabor filter.
    theta: controls the orientation of the Gabor function.
    lambd: is the wavelength of the sinusoidal component - governs the width of the strips of the Gabor function.
    gamma: aspect ratio or gamma controls the height of the Gabor function.
    psi: The phase offset of the sinusoidal function
    """
    gabor_kernel = cv.getGaborKernel((k_size, k_size), sigma, theta, lambd, gamma, psi)
    filtered_image = cv.filter2D(image, cv.CV_8UC3, gabor_kernel)
    return filtered_image


# Parameters for the Gabor filter
ksize = 10  # Filter size
sigma = 1.0  # Gaussian standard deviation
lambd_a = [0.4, 0.5, 0.55]  # List of wavelengths
gamma = 0.5  # Spatial aspect ratio
psi = 0  # Phase offset
angles = np.arange(0, np.pi, np.pi / 4)  # Angles for orientations

# Apply Gabor filters and collect features along with their parameters
gabor_features = []
titles = []  # Store titles for the plots

for lambd in lambd_a:
    for theta in angles:
        gabor_feature = apply_gabor_filter(bbox_cancer_region, ksize, sigma, theta, lambd, gamma, psi)
        gabor_features.append(gabor_feature)
        titles.append(f"lamba: {lambd:.2f}, theta: {theta * (180 / np.pi):.0f} degree")  # Create descriptive titles

# Print feature details
print("Shape of the original image:", bbox_cancer_region.shape, "\n")

for idx, feature in enumerate(gabor_features):
    print(f"Gabor Feature-{idx}- Shape: {feature.shape}, length: {feature.shape[0] * feature.shape[1]}")
    print(feature.reshape(-1))
    print()

"""Visualization Code"""

# Plot the images using plt.subplots
num_features = len(gabor_features)
num_cols = 4  # Number of columns for display
num_rows = int(np.ceil((num_features + 1) / num_cols))  # Convert to integer

fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, num_rows * 3))
axes = axes.ravel()  # Flatten the axes array for easy iteration

# Plot the original bounding box image
axes[0].imshow(bbox_cancer_region, cmap='gray')
axes[0].set_title("Cancer ROI")
axes[0].axis('off')

# Plot each Gabor filtered image with its title
for idx, (filtered_img, title) in enumerate(zip(gabor_features, titles), start=1):
    axes[idx].imshow(filtered_img, cmap='gray')
    axes[idx].set_title(title)
    axes[idx].axis('off')

# Hide any remaining unused axes
for i in range(len(gabor_features) + 1, len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.show()