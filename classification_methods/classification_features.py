from scripts.Dataloader_lesion import BladderCancerDataset, BladderCancerROIDataset
import numpy as np
from skimage.feature import graycomatrix, graycoprops
import pandas as pd


def glcm_feature_extraction(folder_name, roi, time_point):
    """GLCM - range is not between 0 and 1 - no.of features 100 features per image"""
    roi = roi.squeeze().numpy()
    roi = (roi * 255).astype(np.uint8)
    angles = [0, np.pi / 4, np.pi / 2, np.pi * 3 / 4]
    distances = [1, 4, 7]
    # Compute some GLCM properties

    """GLCM Code"""
    features_df = pd.DataFrame()
    for angle in angles:
        for dis in distances:
            glcm_cancer_roi = graycomatrix(
                roi, distances=[dis], angles=[angle], levels=256, symmetric=True, normed=True
                )
            # returns a 2d numpy array - use [0,0] to index single value
            dissimilarity = graycoprops(glcm_cancer_roi, 'dissimilarity')[0, 0]
            correlation = graycoprops(glcm_cancer_roi, 'correlation')[0, 0]
            energy = graycoprops(glcm_cancer_roi, 'energy')[0, 0]
            contrast = graycoprops(glcm_cancer_roi, 'contrast')[0, 0]
            homogeneity = graycoprops(glcm_cancer_roi, 'homogeneity')[0, 0]
            #--------------------------------------------
            features_df[f"dissimilarity:angle-{angle}:distance-{dis}"] = [dissimilarity]
            features_df[f"correlation:angle-{angle}:distance-{dis}"] = [correlation]
            features_df[f"energy:angle-{angle}:distance-{dis}"] = [energy]
            features_df[f"contrast:angle-{angle}:distance-{dis}"] = [contrast]
            features_df[f"homogeneity:angle-{angle}:distance-{dis}"] = [homogeneity]
    features_df.index = [folder_name]
    features_df["type"] = time_point
    return features_df


def get_features():
    extraction_methods = {"HOG": 0, "GLCM": 1, "SIFT": 2, "GLOH": 3, "Fourier_Transform": 4, "Gabor_Filter": 5}
    base_dataset = BladderCancerDataset(
        root_dir='../data/original/Al-Bladder Cancer'
        )
    roi_per_image = 1

    roi_dataset = BladderCancerROIDataset(
        base_dataset,
        roi_width=128,
        roi_height=128,
        overlap=0.25,
        max_rois_per_image=roi_per_image
        )
    cancer_rois_by_folder = roi_dataset.get_cancer_samples()
    cancer_feature_dataframe_list = []

    # Pass cancer roi one by one
    for folder, cancer_roi in cancer_rois_by_folder.items():
        if "-time_point" not in folder:
            # Access the folder name
            folder_name = folder
            # Access the cancer ROI
            roi = cancer_roi
            # Get the corresponding time point using the folder name
            time_point_key = f"{folder}-time_point"
            time_point = cancer_rois_by_folder.get(time_point_key)

            # Now pass these to your function
            features = glcm_feature_extraction(folder_name, roi, time_point)
            cancer_feature_dataframe_list.append(features)
    Dataframe_cancer_feature = pd.concat(cancer_feature_dataframe_list, axis=0)
    Dataframe_cancer_feature.index.name = "patient_id"
    Dataframe_cancer_feature["label"] = 1
    Dataframe_cancer_feature.to_csv("glcm_cancer_features_svm.csv")

    # Iterate over the noncancerous roi dataset and organize ROIs by folder_name
    non_cancer_feature_dataframe_list = []
    for roi in roi_dataset:
        folder_name = roi["ct_folder"]
        time_point = roi["time_point"]
        Roi = roi["image"]
        features = glcm_feature_extraction(folder_name, Roi, time_point)
        non_cancer_feature_dataframe_list.append(features)
    Dataframe_non_cancer_feature = pd.concat(non_cancer_feature_dataframe_list, axis=0)
    Dataframe_non_cancer_feature.index.name = "patient_id"
    Dataframe_non_cancer_feature["label"] = 0
    Dataframe_non_cancer_feature.to_csv("glcm_non_cancer_features_svm.csv")

    return Dataframe_cancer_feature, Dataframe_non_cancer_feature
