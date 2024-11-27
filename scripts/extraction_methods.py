from skimage.feature import hog
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import cv2 as cv
import pandas as pd


# roi range (0,1) and is float32
def feature_extraction_method(rois, argument):
    """

    :param rois: both cancer roi and non-cancer roi's corresponding to a single image
    :param argument: specify the method for feature extraction
    :return: features corresponding to cancer and non caner roi's for a single image as a list
    """
    roi_features = []
    match argument:
        case 0:
            for roi in rois:
                roi = roi.squeeze().numpy()
                roi_features.append(hog_feature_extractor(roi))
            return roi_features
        case 1:
            for roi in rois:
                roi = roi.squeeze().numpy()
                roi = (roi * 255).astype(np.uint8)  # converting to unsigned int for glcm, float is not supported
                roi_features.append(glcm_feature_extractor(roi))
            return roi_features
        case 2:
            for roi in rois:
                roi = roi.squeeze().numpy()
                roi = (roi * 255).astype(np.uint8)  # converting to unsigned int
                roi_features.append(sift_feature_extractor(roi))
            return roi_features
        case 3:
            return 3
        case 4:
            for roi in rois:
                roi = roi.squeeze().numpy()
                roi_features.append(fourier_transform_feature_extraction(roi))
            return roi_features
        case default:
            return "something"


def hog_feature_extractor(image):
    """Hog Descriptor Code - using Hog - max feature length 3240 and min 324  - 1 dimensional vector and range (0,1)"""
    # Calculate HOG features
    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Define a minimum pixel cell size and adjust based on image dimensions
    min_cell_size = 2  # minimum cell size
    scaling_factor = 8  # scale factor to control how many cells to aim for in each dimension

    # Calculate pixels_per_cell based on image size
    pixels_per_cell = (
        max(min_cell_size, height // scaling_factor),
        max(min_cell_size, width // scaling_factor)
        )
    feature, hog_image = hog(image, pixels_per_cell=pixels_per_cell, cells_per_block=(2, 2), visualize=True)
    min_val = np.min(feature)
    max_val = np.max(feature)
    print(f"Range of roi values: min = {min_val}, max = {max_val}")
    return feature


def glcm_feature_extractor(roi):
    """GLCM - range is not between 0 and 1 - no.of features 100 features per image"""
    angles = [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4]
    distances = [1, 2, 3, 4, 5]
    # Compute some GLCM properties
    dissimilarity = []
    correlation = []
    energy = []
    contrast = []
    homogeneity = []

    """GLCM Code"""
    angles_column = []
    distances_column = []
    features_df = pd.DataFrame()
    for angle in angles:
        for dis in distances:
            angles_column.append(angle)
            distances_column.append(dis)
            glcm_cancer_roi = graycomatrix(
                roi, distances=[dis], angles=[angle], levels=256, symmetric=True, normed=True
                )
            # returns a 2d numpy array - use [0,0] to index single value
            dissimilarity.append(graycoprops(glcm_cancer_roi, 'dissimilarity')[0, 0])
            correlation.append(graycoprops(glcm_cancer_roi, 'correlation')[0, 0])
            energy.append(graycoprops(glcm_cancer_roi, 'energy')[0, 0])
            contrast.append(graycoprops(glcm_cancer_roi, 'contrast')[0, 0])
            homogeneity.append(graycoprops(glcm_cancer_roi, 'homogeneity')[0, 0])
    # return dissimilarity + correlation + energy + contrast + homogeneity
    features_df["angle"] = angles_column
    features_df["distance"] = distances_column
    features_df["dissimilarity"] = dissimilarity
    features_df["correlation"] = correlation
    features_df["energy"] = energy
    features_df["contrast"] = contrast
    features_df["homogeneity"] = homogeneity
    return features_df


def fourier_transform_feature_extraction(image):
    """Max feature length 16384 and min 81"""
    DFT = cv.dft(np.float32(image), flags=cv.DFT_COMPLEX_OUTPUT)

    # reposition the zero-frequency component to the spectrum's middle
    fourier_shift = np.fft.fftshift(DFT)

    # calculate the magnitude of the Fourier Transform - intensity of each frequency component
    magnitude = 20 * np.log(cv.magnitude(fourier_shift[:, :, 0], fourier_shift[:, :, 1]))
    return magnitude.reshape(-1)


def sift_feature_extractor(image):
    """128 features per keypoint"""
    # nfeatures - maximum best features if 0 return all,keep contrastThreshold low to extract small details,
    # edgeThreshold high to detect edges
    sift = cv.SIFT_create(nfeatures=0, nOctaveLayers=4, contrastThreshold=0.001, edgeThreshold=10)
    keypoints, descriptors = sift.detectAndCompute(image, None)  # keypoints and corresponding descriptors
    if len(keypoints) == 0:
        return [0]
    return descriptors.reshape(-1)
