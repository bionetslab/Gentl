from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt
import dummy_roi_extractor


# Call the function and get the bounding box region of the image
bbox_cancer_region = dummy_roi_extractor.extract_cancer_roi()

"""Hog Descriptor Code"""
# Calculate HOG features - features are normalised between 0 and 1
features, hog_image = hog(bbox_cancer_region, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

"""Visualization Code"""
# Display the original image and HOG features

fig, axes = plt.subplots(1, 3, figsize=(10, 4))

axes[0].imshow(bbox_cancer_region, cmap='gray')
axes[0].set_title(f"Bounding Box Image")
axes[0].axis('off')

axes[1].imshow(hog_image, cmap='gray')
axes[1].set_title('Hog Feature Image')
axes[1].axis('off')

axes[2].imshow(hog_image_rescaled, cmap='gray')
axes[2].set_title('Hog Feature Enhanced Image')
axes[2].axis('off')

plt.tight_layout()
plt.show()

