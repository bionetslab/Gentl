import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
import cv2 as cv
import numpy as np

# Read the image
original_image = cv.imread('cancer.jpg')
gray_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY)


Roi_size = 30

cancer_locations = [(186, 198),(220,198)]  # (x,y) coordinate points
# normal_locations = [(150, 141),(315, 357),(283,165)]
normal_locations = [(155,150),(315, 357)]
cancer_roi = []
normal_roi = []

# Extract cancer roi from the image
for location in cancer_locations:
    cancer_roi.append(gray_image[location[0]:location[0] + Roi_size, location[1]:location[1] + Roi_size])

for location in normal_locations:
    normal_roi.append(gray_image[location[0]:location[0] + Roi_size, location[1]:location[1] + Roi_size])

# compute some GLCM properties each patch
dis = []
cor = []
energy = []
contrast = []
homogeneity = []

"""GLCM Code"""
for patch in cancer_roi + normal_roi:
    glcm = graycomatrix(
        patch, distances=[1], angles=[np.pi/2], levels=256, symmetric=True, normed=True
        )
    dis.append(graycoprops(glcm, 'dissimilarity')[0, 0])
    cor.append(graycoprops(glcm, 'correlation')[0, 0])
    energy.append(graycoprops(glcm, 'energy')[0])
    contrast.append(graycoprops(glcm, 'contrast')[0])
    homogeneity.append(graycoprops(glcm, 'homogeneity')[0])


"""Visualization Code"""
# create the figure
print(dis)
print(cor)
fig = plt.figure(figsize=(8, 8))

# display original image with locations of patches
ax = fig.add_subplot(3, 2, 1)
ax.imshow(gray_image, cmap=plt.cm.gray, vmin=0, vmax=255)
for x, y in cancer_locations:
    ax.plot(x + Roi_size / 2, y + Roi_size / 2, 'gs')
for x, y in normal_locations:
    ax.plot(x + Roi_size / 2, y + Roi_size / 2, 'bs')
ax.set_xlabel('Original Image')
ax.axis('image')

# for each patch, plot (dissimilarity, correlation)
ax = fig.add_subplot(3, 2, 2)
ax.plot(dis[: len(cancer_roi)], cor[: len(cancer_roi)], 'go', label='Cancer')
ax.plot(dis[len(cancer_roi) :], cor[len(cancer_roi) :], 'bo', label='Normal')

# Annotate each point with its name
for i, (x, y) in enumerate(zip(dis, cor)):
    ax.text(x, y, i+1, fontsize=10, ha='right')

ax.set_xlabel('GLCM Dissimilarity')
ax.set_ylabel('GLCM Correlation')
ax.legend()

# display the image patches
for i, patch in enumerate(cancer_roi):
    ax = fig.add_subplot(3, len(cancer_roi), len(cancer_roi) * 1 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray, vmin=0, vmax=255)
    ax.set_xlabel(f"Cancer {i + 1}")

for i, patch in enumerate(normal_roi):
    ax = fig.add_subplot(3, len(normal_roi), len(normal_roi) * 2 + i + 1)
    ax.imshow(patch, cmap=plt.cm.gray, vmin=0, vmax=255)
    ax.set_xlabel(f"Normal {i + 1}")

fig.suptitle('Grey level co-occurrence matrix features', fontsize=14, y=1.05)
plt.tight_layout()
plt.show()