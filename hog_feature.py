import cv2 as cv
from skimage.feature import hog
from skimage import exposure
import matplotlib.pyplot as plt

# Load an image
image = cv.imread('cancer.jpg', cv.IMREAD_GRAYSCALE)

"""Hog Descriptor Code"""
# Calculate HOG features - features are normalised between 0 and 1
features, hog_image = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
# for i in features:
#     print(i)
# Rescale HOG features for better visualization
hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

"""Visualization Code"""
# Display the original image and HOG features
plt.figure(figsize=(8, 2))

# First subplot for the original image
plt.subplot(131)  # Use 131 for the first subplot (1 row, 3 columns, 1st subplot)
plt.imshow(image, cmap=plt.cm.gray)
plt.axis('off')
plt.title('Original Image')

# Second subplot for the HOG image before rescaling
plt.subplot(132)  # Use 132 for the second subplot (1 row, 3 columns, 2nd subplot)
plt.imshow(hog_image, cmap=plt.cm.gray)
plt.axis('off')
plt.title('Before Rescale HOG Image')

# Third subplot for the HOG image after rescaling
plt.subplot(133)  # Use 133 for the third subplot (1 row, 3 columns, 3rd subplot)
plt.imshow(hog_image_rescaled, cmap=plt.cm.gray)
plt.axis('off')
plt.title('HOG Features')

plt.show()

