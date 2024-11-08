from scripts.Dataloader_lesion import BladderCancerDataset, BladderCancerROIDataset
from scripts.extraction_methods import feature_extraction_method


def extract_features():
    """
    Call this function to extract features from images - roi_per_image can be passed as an argument
    :return:
    features returned as a dictionary of list-with key as folder name and value as a list of features
    {"CT-1": [cancer_roi_feature,non-cancer_roi1_feature,non-cancer_roi2_feature...],"CT-2"[..]...}
    """
    extraction_methods = {"HOG": 0, "GLCM": 1, "SIFT": 2, "GLOH": 3, "Fourier_Transform": 4, "Gabor_Filter": 5}
    selected_method = "SIFT" # select the method for feature extraction
    base_dataset = BladderCancerDataset(
        root_dir=r'C:\Users\vinee\OneDrive\Desktop\Project_Gentl\Gentl\data\original\Al-Bladder Cancer'
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


extracted_features = extract_features()
s = []
for i in extracted_features:
    for j in extracted_features[i]:
        print(i,j)
        s.append(len(j))
print(max(s))


print(min(s))





























