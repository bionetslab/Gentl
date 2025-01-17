import numpy as np
import pandas as pd


def get_gentl_features_from_csv(selected_feature, max_no_of_rois, gentl_result_param):
    """
    Fetch the gentl result csv and add labels

    Arguments:
    selected_feature: glcm features
    max_no_of_rois: no of extracted rois
    gentl_result_param: average mean distance,best distance or average generation

    Returns:
    Dataframe with labels marked
    """
    gentl_dataframe = pd.read_csv(
        f"../glcm_bladder_average_gentl_results/{max_no_of_rois}/{gentl_result_param}/"
        f"{selected_feature}_{gentl_result_param}_{max_no_of_rois}_rois.csv"
        )
    patient_count = len(gentl_dataframe['patient_id'])
    gentl_dataframe["label"] = list(map(int, np.ones(patient_count, dtype=int)))
    return gentl_dataframe


def get_features_from_csv(selected_feature, max_no_of_rois):
    max_no_of_rois = max_no_of_rois  # can be set to 10,20,30,40,50
    selected_feature = selected_feature  # [dissimilarity,correlation,energy,contrast,homogeneity]
    features_per_roi = 20  # 20 dissimilarity features per roi
    random_selection = False  # select one normal region feature from 1 to 20
    feature_df = pd.read_csv(
        f'../scripts/extracted_glcm_features/{max_no_of_rois}/glcm_{selected_feature}_features_{max_no_of_rois}_rois.csv'
        )

    # Get a list of unique patient IDs
    patient_ids = feature_df['patient_id'].unique()

    # Create an empty dataframe to store cancer_df
    cancer_features_df = pd.DataFrame()
    healthy_features_df = pd.DataFrame()

    # Get data for each patient using patient id
    for patient_id in patient_ids:
        patient_data = feature_df[feature_df['patient_id'] == patient_id]

        # Extract the selected features for the cancer region
        cancer_data = patient_data[patient_data['roi_condition'] == 'cancer']
        cancer_features = cancer_data[selected_feature].values

        # Dynamically generate column names for the features
        cancer_feature_columns = {f'{selected_feature}_{i + 1}': cancer_features[i] for i in
                                  range(len(cancer_features))}

        # Prepare a row for the DataFrame
        cancer_new_row = {
            'patient_id': patient_id,
            'label': 1  # Cancer label
            }
        # Add feature columns to the row
        cancer_new_row.update(cancer_feature_columns)

        # Append the new row to the DataFrame
        cancer_features_df = pd.concat([cancer_features_df, pd.DataFrame([cancer_new_row])])

        # Extract the selected features for the healthy region
        healthy_data = patient_data[patient_data['roi_condition'] == 'healthy']
        healthy_features = healthy_data[selected_feature].values

        # Randomly select a healthy roi from max roi number or set a number eg 5
        random_feature_set_index = np.random.randint(0, max_no_of_rois) if random_selection else 3
        start_index = random_feature_set_index * features_per_roi
        end_index = start_index + features_per_roi

        # Extract the subsequent 20 features from the selected roi
        selected_healthy_features = healthy_features[start_index:end_index]

        # Dynamically generate column names for the features
        healthy_feature_columns = {f'{selected_feature}_{i + 1}': selected_healthy_features[i] for i in
                                   range(len(selected_healthy_features))}

        # Prepare a row for the DataFrame
        healthy_new_row = {
            'patient_id': patient_id,
            'label': 0  # Healthy label
            }
        # Add feature columns to the row
        healthy_new_row.update(healthy_feature_columns)

        # Append the new row to the DataFrame
        healthy_features_df = pd.concat([healthy_features_df, pd.DataFrame([healthy_new_row])])
    return cancer_features_df, healthy_features_df


def get_features_by_invasion(selected_feature, max_no_of_rois, gentl_result_param, gentl_flag):
    """
    Merge cancer features to include label for NMIBC(0) {Ta,Tis,T1} and MIBC(1) {T2,T3,T4}

    Returns:
    Dataframe_cancer_with_types: A dataframe with patient IDs,
     cancer features, and binary cancer type labels (0 for NMIBC, 1 for MIBC).
    """
    # -------------------NMIBC Vs MIBC----------------------
    if gentl_flag:
        cancer_features_df = get_gentl_features_from_csv(selected_feature, max_no_of_rois, gentl_result_param)
    else:
        cancer_features_df, _ = get_features_from_csv(selected_feature, max_no_of_rois)
    Dataframe_cancer_with_stages = filter_T0_based_on_flag(cancer_features_df)
    Dataframe_cancer_with_stages["cancer_invasion_label"] = Dataframe_cancer_with_stages["cancer_stage"].map(
        {"Ta": 0, "Tis": 0, "T1": 0, "T2": 1, "T3": 1, "T4": 1}
        ).astype(int)
    # Dataframe_cancer_with_stages.to_csv(
    #     f"./features/{max_no_of_rois}/gentl_cancer_{selected_feature}_features_{max_no_of_rois}_rois_with_invasion.csv"
    #     )
    return Dataframe_cancer_with_stages


def get_features_by_stage(selected_feature, max_no_of_rois,gentl_result_param, gentl_flag):
    """
    Merge cancer features to include label for different stages

    Returns:
    Dataframe_cancer_with_types: A dataframe with patient IDs, cancer features, and label for cancer stage.
    """
    # -------------------T0 Vs Ta Vs Tis Vs T1 Vs T2 Vs T3 Vs T4----------------------
    if gentl_flag:
        cancer_features_df = get_gentl_features_from_csv(selected_feature, max_no_of_rois, gentl_result_param)
    else:
        cancer_features_df, _ = get_features_from_csv(selected_feature, max_no_of_rois)
    Dataframe_cancer_with_stages = filter_T0_based_on_flag(cancer_features_df, False)
    stage_mapping = {
        "T0": 0, "Ta": 1, "Tis": 2, "T1": 3, "T2": 4, "T3": 5, "T4": 6
        }
    Dataframe_cancer_with_stages["cancer_stage_label"] = Dataframe_cancer_with_stages["cancer_stage"].map(
        stage_mapping
        ).astype(int)
    # Dataframe_cancer_with_stages.to_csv(
    #     f"./features/{max_no_of_rois}/glcm_cancer_{selected_feature}_features_{max_no_of_rois}_rois_with_stages.csv"
    #     )
    return Dataframe_cancer_with_stages


def get_all_features(selected_feature, max_no_of_rois):
    """
    Merge cancer features(Ta,Tis,T1,T2,T3,T4) and healthy features (T0,Ta,Tis,T1,T2,T3,T4)

    Returns:
    Dataframe_cancer_with_types: A dataframe with patient IDs, cancer and healthy features with labels
    """
    # -------------------Cancer Vs Non-cancer-----------------------------------------
    cancer_features_df, healthy_features_df = get_features_from_csv(selected_feature, max_no_of_rois)
    healthy_features_dataframe = filter_T0_based_on_flag(healthy_features_df, False)
    cancer_features_dataframe = filter_T0_based_on_flag(cancer_features_df)
    full_features_dataframe = pd.concat([cancer_features_dataframe, healthy_features_dataframe], axis=0)
    # full_features_dataframe.to_csv(
    #     f"./features/{max_no_of_rois}/glcm_all_{selected_feature}_features_{max_no_of_rois}_rois.csv"
    #     )
    return full_features_dataframe


def get_early_late_stage_features(selected_feature, max_no_of_rois,gentl_result_param, gentl_flag):
    if gentl_flag:
        cancer_features_df = get_gentl_features_from_csv(selected_feature, max_no_of_rois, gentl_result_param)
    else:
        cancer_features_df, _ = get_features_from_csv(selected_feature, max_no_of_rois)
    Dataframe_cancer_with_stages = filter_T0_based_on_flag(cancer_features_df)
    stage_mapping = {
        "Ta": 0, "Tis": 0,
        "T1": 1, "T2": 1, "T3": 1, "T4": 1
        }
    Dataframe_cancer_with_stages["cancer_stage_label"] = Dataframe_cancer_with_stages["cancer_stage"].map(
        stage_mapping
        ).astype(int)
    # Dataframe_cancer_with_stages.to_csv(
    #     f"./features/{max_no_of_rois}/glcm_cancer_{selected_feature}_features_{max_no_of_rois}_rois_with_early_late_stages.csv"
    #     )
    return Dataframe_cancer_with_stages


def get_features_ptc_vs_mibc(selected_feature, max_no_of_rois,gentl_result_param, gentl_flag):
    """
    Retrieves features for Post-Treatment Changes (PTC) [T0] versus Muscle-Invasive Bladder Cancer (MIBC) [T2,T3,T4]

    """
    if gentl_flag:
        cancer_features_df = get_gentl_features_from_csv(selected_feature, max_no_of_rois, gentl_result_param)
    else:
        cancer_features_df, _ = get_features_from_csv(selected_feature, max_no_of_rois)
    Dataframe_features = filter_T0_based_on_flag(cancer_features_df, False)

    Dataframe_features = Dataframe_features.loc[
        Dataframe_features['cancer_stage'].isin(['T0', 'T2', 'T3', 'T4'])]

    stage_mapping = {
        "T0": 0, "T2": 1, "T3": 1, "T4": 1
        }
    Dataframe_features["cancer_stage_label"] = Dataframe_features["cancer_stage"].map(
        stage_mapping
        ).astype(int)
    # Dataframe_features.to_csv(
    #     f"./features/{max_no_of_rois}/glcm_cancer_{selected_feature}_features_{max_no_of_rois}_rois_with_ptc_vs_mibc.csv"
    #     )

    return Dataframe_features


def filter_T0_based_on_flag(features_dataframe, filter_flag=True, ):
    """
    Filters out the T0 cancer type from the dataframe if filter is True else return dataframe with all stages

    Arguments:
    features_dataframe: Dataframe to filter out
    filter_flag: if True remove T0

    Returns:
    cancer_df {Dataframe} - without T0 stage
    """
    csv_path = '../data/original/Al-Bladder Cancer/Data_CT only with anonymized ID 11-13-24_clean.csv'
    df_cancer_stages = pd.read_csv(csv_path)

    Dataframe_with_stages = pd.merge(
        features_dataframe, df_cancer_stages[["Final Path", "Anonymized ID"]], left_on='patient_id',
        right_on="Anonymized ID", how='left'
        )

    Dataframe_with_stages = Dataframe_with_stages.rename(
        columns={"Final Path": "cancer_stage"}
        )
    Dataframe_with_stages = Dataframe_with_stages.drop("Anonymized ID", axis=1)
    Dataframe_with_stages = Dataframe_with_stages.set_index("patient_id")
    if filter_flag:
        Dataframe_with_stages = Dataframe_with_stages.loc[
            Dataframe_with_stages['cancer_stage'] != 'T0']

    return Dataframe_with_stages


def get_tasks():
    tasks = ["Classification of cancer invasion - NMIBC vs MIBC ", "Classification of lesion vs control",
             "Classification of cancer stages - TA vs Tis vs T1 vs T2 vs T3 vs T4",
             "Classification of early vs late stage", "Classification of post treatment changes vs MIBC"]
    return tasks

# if __name__ == "__main__":
# get_features_by_invasion(selected_feature="dissimilarity", max_no_of_rois=10)
# get_features_by_stage(selected_feature="dissimilarity", max_no_of_rois=10)
# get_all_features(selected_feature="dissimilarity", max_no_of_rois=10)
# get_early_late_stage_features(selected_feature="dissimilarity", max_no_of_rois=10)
# get_features_ptc_vs_mibc(selected_feature="dissimilarity", max_no_of_rois=10)
