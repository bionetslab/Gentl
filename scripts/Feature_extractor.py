from scripts.Dataloader_lesion import BladderCancerDataset, BladderCancerROIDataset
from scripts.extraction_methods import feature_extraction_method
import pandas as pd


def extract_features(selected_method, roi_per_image=10):
    """
    Function to extract features from images based on the selected feature extraction method

    Arguments:
    selected_method {string}: Selected feature extraction method - GLCM

    Returns:
    feature_results {dictionary}: key as folder name and value as list of features
    {"CT-1": [cancer_roi_feature,non-cancer_roi1_feature,non-cancer_roi2_feature...],"CT-2"[..]...}
    """
    extraction_methods = {"HOG": 0, "GLCM": 1, "SIFT": 2, "GLOH": 3, "Fourier_Transform": 4, "Gabor_Filter": 5}
    base_dataset = BladderCancerDataset(
        root_dir='../data/original/Al-Bladder Cancer'
        )
    roi_per_image = roi_per_image

    roi_dataset = BladderCancerROIDataset(
        base_dataset,
        roi_width=8,
        roi_height=8,
        overlap=0.25,
        max_rois_per_image=roi_per_image
        )

    cancer_rois_dataset = roi_dataset.get_cancer_samples()
    feature_results = {}

    # Iterate over cancer roi's and organise ROI by folder_name
    cancer_rois_by_folder = {}
    """eg:{'CT-009':[ROI],'CT-010':[ROI]..}"""
    for roi in cancer_rois_dataset:
        folder_name = roi['ct_folder']
        if folder_name not in cancer_rois_by_folder:
            cancer_rois_by_folder[folder_name] = []
        cancer_rois_by_folder[folder_name].append(roi['image'])

    # Iterate over the noncancerous roi dataset and organize ROIs by folder_name
    """eg:{'CT-009':[ROI1,ROI2],'CT-010':[ROI1,ROI2]}"""
    roi_grouped_by_folder = {}
    for roi in roi_dataset:
        folder_name = roi['ct_folder']
        if folder_name not in roi_grouped_by_folder:
            roi_grouped_by_folder[folder_name] = []
        roi_grouped_by_folder[folder_name].append(roi['image'])  # Add each non-cancer ROI

    # Combine each cancer ROI with its non-cancer ROIs and pass them to the feature extraction method
    for folder_name, non_cancer_rois in roi_grouped_by_folder.items():  # Extract: key and values
        if folder_name in cancer_rois_by_folder:
            # Create a list starting with the cancer ROI, followed by all non-cancer ROIs for this folder
            """eg:[C_ROI,ROI1,ROI2]"""
            roi_list = cancer_rois_by_folder[folder_name] + non_cancer_rois
            # Pass this ROIs list to the feature extraction method
            features = feature_extraction_method(roi_list, extraction_methods[selected_method])
            feature_results[folder_name] = features

    return feature_results


if __name__ == "__main__":

    max_roi_per_image = 30  # select the number of rois to extract per image
    selected_method = "GLCM"  # select the method for feature extraction
    extracted_features = extract_features(selected_method, max_roi_per_image)  # calls the extract features method
    extracted_features_df = pd.DataFrame(extracted_features).T

    extracted_features_df = extracted_features_df.rename(columns={0: "cancer"})
    cols_ = list(extracted_features_df.columns)
    no_of_columns = len(cols_)
    if no_of_columns > 1:
        columns_rename_dict = {}
        for col in cols_[1:]:
            columns_rename_dict[col] = f"healthy_{col}"
        extracted_features_df = extracted_features_df.rename(columns=columns_rename_dict)

    cols_names = list(extracted_features_df.columns)
    Dissimilarity = []
    Correlation = []
    Energy = []
    Contrast = []
    Homogeneity = []

    if selected_method == "GLCM":
        for index, row in extracted_features_df.iterrows():
            _dissimilarity = []
            _correlation = []
            _energy = []
            _contrast = []
            _homogeneity = []
            # -----------------------
            for col in cols_names:  # cols_ = ["cancer",healthy_1,healthy_2,....]
                _dissimilarity_ = row.loc[col][["angle", "distance", "dissimilarity"]].copy()
                _dissimilarity_["roi_condition"] = "cancer" if col == "cancer" else "healthy"
                _dissimilarity_["patient_id"] = index
                _correlation_ = row.loc[col][["angle", "distance", "correlation"]].copy()
                _correlation_["roi_condition"] = "cancer" if col == "cancer" else "healthy"
                _correlation_["patient_id"] = index
                _energy_ = row.loc[col][["angle", "distance", "energy"]].copy()
                _energy_["roi_condition"] = "cancer" if col == "cancer" else "healthy"
                _energy_["patient_id"] = index
                _contrast_ = row.loc[col][["angle", "distance", "contrast"]].copy()
                _contrast_["roi_condition"] = "cancer" if col == "cancer" else "healthy"
                _contrast_["patient_id"] = index
                _homogeneity_ = row.loc[col][["angle", "distance", "homogeneity"]].copy()
                _homogeneity_["roi_condition"] = "cancer" if col == "cancer" else "healthy"
                _homogeneity_["patient_id"] = index

                _dissimilarity.append(_dissimilarity_)  # append each dataframe to the list
                _correlation.append(_correlation_)
                _energy.append(_energy_)
                _contrast.append(_contrast_)
                _homogeneity.append(_homogeneity_)

            dissimilarity = pd.concat(
                _dissimilarity, axis=0
                )  # axis=0, stacks vertically on top of each other for each patient
            correlation = pd.concat(_correlation, axis=0)
            energy = pd.concat(_energy, axis=0)
            contrast = pd.concat(_contrast, axis=0)
            homogeneity = pd.concat(_homogeneity, axis=0)

            Dissimilarity.append(dissimilarity)  # first element belongs to one image ie 3 regions here
            Correlation.append(correlation)
            Energy.append(energy)
            Contrast.append(contrast)
            Homogeneity.append(homogeneity)
    # ---
    Dissimilarity = pd.concat(Dissimilarity, axis=0)  # increase rows - on top of each other
    Correlation = pd.concat(Correlation, axis=0)
    Energy = pd.concat(Energy, axis=0)
    Contrast = pd.concat(Contrast, axis=0)
    Homogeneity = pd.concat(Homogeneity, axis=0)
    # ---
    Dissimilarity.to_csv(
        f"./extracted_glcm_features/{max_roi_per_image}/glcm_dissimilarity_features_{max_roi_per_image}_rois.csv"
        )
    Correlation.to_csv(
        f"./extracted_glcm_features/{max_roi_per_image}/glcm_correlation_features_{max_roi_per_image}_rois.csv"
        )
    Energy.to_csv(f"./extracted_glcm_features/{max_roi_per_image}/glcm_energy_features_{max_roi_per_image}_rois.csv")
    Contrast.to_csv(
        f"./extracted_glcm_features/{max_roi_per_image}/glcm_contrast_features_{max_roi_per_image}_rois.csv"
        )
    Homogeneity.to_csv(
        f"./extracted_glcm_features/{max_roi_per_image}/glcm_homogeneity_features_{max_roi_per_image}_rois.csv"
        )
