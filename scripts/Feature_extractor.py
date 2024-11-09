from scripts.Dataloader_lesion import BladderCancerDataset, BladderCancerROIDataset
from scripts.extraction_methods import feature_extraction_method
import pandas as pd

def extract_features(selected_method):
    """
    Call this function to extract features from images - roi_per_image can be passed as an argument
    :return:
    features returned as a dictionary of list-with key as folder name and value as a list of features
    {"CT-1": [cancer_roi_feature,non-cancer_roi1_feature,non-cancer_roi2_feature...],"CT-2"[..]...}
    """
    extraction_methods = {"HOG": 0, "GLCM": 1, "SIFT": 2, "GLOH": 3, "Fourier_Transform": 4, "Gabor_Filter": 5}
    base_dataset = BladderCancerDataset(
        root_dir='../data/original/Al-Bladder Cancer'
        )
    roi_per_image = 2

    roi_dataset = BladderCancerROIDataset(
        base_dataset,
        roi_width=128,
        roi_height=128,
        overlap=0.25,
        max_rois_per_image=roi_per_image
        )
    cancer_rois_by_folder = roi_dataset.get_cancer_samples()
    feature_results = {}

    # Iterate over the noncancerous roi dataset and organize ROIs by folder_name
    """eg:{'CT-009':[ROI1,ROI2],'CT-010':[ROI1,ROI2]}"""
    roi_grouped_by_folder = {}
    for roi in roi_dataset:
        folder_name = roi['ct_folder']
        if folder_name not in roi_grouped_by_folder:
            roi_grouped_by_folder[folder_name] = []
        roi_grouped_by_folder[folder_name].append(roi['image'])  # Add each non-cancer ROI

    # Combine each cancer ROI with its non-cancer ROIs and pass them to the feature extraction method
    for folder_name, non_cancer_rois in roi_grouped_by_folder.items(): # Extract: key and values
        if folder_name in cancer_rois_by_folder:
            # Create a list starting with the cancer ROI, followed by all non-cancer ROIs for this folder
            """eg:[C_ROI,ROI1,ROI2]"""
            roi_list = [cancer_rois_by_folder[folder_name]] + non_cancer_rois
            # Pass this list to the feature extraction method
            features = feature_extraction_method(roi_list, extraction_methods[selected_method])
            feature_results[folder_name] = features

    return feature_results


selected_method = "GLCM" # select the method for feature extraction
extracted_features = extract_features(selected_method)
extracted_features_df=pd.DataFrame(extracted_features).T
extracted_features_df=extracted_features_df.rename(columns={0:"cancer"})
cols_=list(extracted_features_df.columns)
no_of_columns=len(cols_)
if no_of_columns>1:
    columns_rename_dict={}
    for col in cols_[1:]:
        columns_rename_dict[col]=f"healthy_{col}"
    extracted_features_df=extracted_features_df.rename(columns=columns_rename_dict)

Dissimilarity=[]
Correlation=[]
Energy=[]
Contrast=[]
Homogeneity=[]

if selected_method=="GLCM":
    for index, row in extracted_features_df.iterrows():
        _dissimilarity = []
        _correlation = []
        _energy = []
        _contrast = []
        _homogeneity = []
        # -----------------------
        _dissimilarity_ = row["cancer"][["angle", "distance", "dissimilarity"]]
        _correlation_ = row["cancer"][["angle", "distance", "correlation"]]
        _energy_ = row["cancer"][["angle", "distance", "energy"]]
        _contrast_ = row["cancer"][["angle", "distance", "contrast"]]
        _homogeneity_ = row["cancer"][["angle", "distance", "homogeneity"]]
        # -----------------------
        if no_of_columns==1:
            _dissimilarity = _dissimilarity_
            _dissimilarity["roi_condition"] = "cancer"
            _dissimilarity["patient_id"] = index
            _correlation = _correlation_
            _energy = _energy_
            _contrast = _contrast_
            _homogeneity = _homogeneity_

        else:
            count = -1
            for col in cols_:
                count+=1
                if count==0:
                    multiple_col_flag=1 # No action.
                else:
                    _dissimilarity_ = row[col][["angle", "distance", "dissimilarity"]]
                    _dissimilarity_["roi_condition"]="healthy"
                    _dissimilarity_["patient_id"] = index
                    _correlation_ = row[col][["angle", "distance", "correlation"]]
                    _energy_ = row[col][["angle", "distance", "energy"]]
                    _contrast_ = row[col][["angle", "distance", "contrast"]]
                    _homogeneity_ = row[col][["angle", "distance", "homogeneity"]]

                _dissimilarity.append(_dissimilarity_)
                _correlation.append(_correlation_)
                _energy.append(_energy_)
                _contrast.append(_contrast_)
                _homogeneity.append(_homogeneity_)

        dissimilarity=pd.concat(_dissimilarity, axis=0)
        correlation=pd.concat(_correlation, axis=0)
        energy=pd.concat(_energy, axis=0)
        contrast=pd.concat(_contrast, axis=0)
        homogeneity=pd.concat(_homogeneity, axis=0)

        Dissimilarity.append(dissimilarity)
        Correlation.append(correlation)
        Energy.append(energy)
        Contrast.append(contrast)
        Homogeneity.append(homogeneity)
# ---
Dissimilarity=pd.concat(Dissimilarity, axis=0)
Correlation=pd.concat(Correlation, axis=0)
Energy=pd.concat(Energy, axis=0)
Contrast=pd.concat(Contrast, axis=0)
Homogeneity=pd.concat(Homogeneity, axis=0)
# ---
Dissimilarity.to_csv("glcm_dissimilarity_features.csv")
Correlation.to_csv("glcm_correlation_features.csv")
Energy.to_csv("glcm_energy_features.csv")
Contrast.to_csv("glcm_contrast_features.csv")
Homogeneity.to_csv("glcm_homogeneity_features.csv")