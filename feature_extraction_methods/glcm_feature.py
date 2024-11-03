import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
import numpy as np
import dummy_roi_extractor

# Call the function and get the bounding box region of the image
bbox_cancer_region_image = dummy_roi_extractor.extract_cancer_roi()
bbox_non_cancer_region_image, _ = dummy_roi_extractor.extract_non_cancer_rois()

angles = [0, np.pi / 4, np.pi / 2, np.pi * 3 / 2]
# compute some GLCM properties each patch
dis = []
cor = []
energy = []
contrast = []
homogeneity = []


"""GLCM Code"""
glcm_cancer_roi = graycomatrix(
    bbox_cancer_region_image, distances=[5], angles=[np.pi / 2], levels=256, symmetric=True, normed=True
    )
dis.append(graycoprops(glcm_cancer_roi, 'dissimilarity')[0, 0])
cor.append(graycoprops(glcm_cancer_roi, 'correlation')[0, 0])
energy.append(graycoprops(glcm_cancer_roi, 'energy')[0])
contrast.append(graycoprops(glcm_cancer_roi, 'contrast')[0])
homogeneity.append(graycoprops(glcm_cancer_roi, 'homogeneity')[0])

for patch in bbox_non_cancer_region_image:
    glcm_non_cancer_roi = graycomatrix(
        patch, distances=[5], angles=[np.pi / 2], levels=256, symmetric=True, normed=True
        )
    dis.append(graycoprops(glcm_non_cancer_roi, 'dissimilarity')[0, 0])
    cor.append(graycoprops(glcm_non_cancer_roi, 'correlation')[0, 0])
    energy.append(graycoprops(glcm_non_cancer_roi, 'energy')[0])
    contrast.append(graycoprops(glcm_non_cancer_roi, 'contrast')[0])
    homogeneity.append(graycoprops(glcm_non_cancer_roi, 'homogeneity')[0])

"""Visualization Code"""
fig, axes = plt.subplots(1, 1, figsize=(6, 4))


axes.plot(dis[0], cor[0], 'ro', label='Cancer Region')
axes.plot(dis[1:], cor[1:], 'bo', label='Non-Cancer Region')
axes.axis('on')

# Set labels, title, and grid
axes.set_xlabel('Dissimilarity (Distance)')
axes.set_ylabel('Correlation')
axes.set_title(f'Cancer vs Non-Cancer')
axes.grid(True)

# Adding a single legend for the figure
handles, labels = axes.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')

plt.tight_layout()
plt.show()
