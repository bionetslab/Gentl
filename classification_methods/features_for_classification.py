import numpy as np
import pandas as pd

max_no_of_rois = 50  # can be set to 10,20,30,50,60
selected_feature = "correlation" #[dissimilarity,correlation,energy,contrast,homogeneity]
features_per_roi = 20  # 20 dissimilarity features per roi
random_selection = False
feature_df = pd.read_csv(f'../glcm_extracted_features_results/glcm_{selected_feature}_features_{max_no_of_rois}_rois.csv')

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
    cancer_feature_columns = {f'{selected_feature}_{i + 1}': cancer_features[i] for i in range(len(cancer_features))}

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
    random_feature_set_index = np.random.randint(0, max_no_of_rois) if random_selection else 10
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


def get_features_by_type():
    """
    Merge cancer features to include label for NMIBC(0) and MIBC(1)

    Returns:
    Dataframe_cancer_with_types: A dataframe with patient IDs, cancer features, and binary cancer type labels (0 for NMIBC, 1 for MIBC).
    """
    # -------------------NMIBC Vs MIBC----------------------
    csv_path = '../data/original/Al-Bladder Cancer/Data_CT only with anonymized ID 11-13-24_clean.csv'  # csv with cancer types
    df_cancer_types = pd.read_csv(csv_path)

    Dataframe_cancer_with_types = pd.merge(
        cancer_features_df, df_cancer_types[["Final Path", "Anonymized ID"]], left_on='patient_id',
        right_on="Anonymized ID", how='left'
        )

    Dataframe_cancer_with_types = Dataframe_cancer_with_types.rename(
        columns={"Final Path": "cancer_type"}
        )
    Dataframe_cancer_with_types = Dataframe_cancer_with_types.drop("Anonymized ID", axis=1)
    Dataframe_cancer_with_types = Dataframe_cancer_with_types.set_index("patient_id")
    Dataframe_cancer_with_types = Dataframe_cancer_with_types.loc[Dataframe_cancer_with_types['cancer_type'] != 'T0']
    Dataframe_cancer_with_types["cancer_type_label"] = Dataframe_cancer_with_types["cancer_type"].map(
        {"Ta": 0, "Tis": 0, "T1": 0, "T2": 1, "T3": 1, "T4": 1}
        ).astype(int)
    Dataframe_cancer_with_types.to_csv(f"./features/glcm_cancer_{selected_feature}_features_{max_no_of_rois}_rois_with_types.csv")
    return Dataframe_cancer_with_types


def get_features_by_sub_type():
    """
    Merge cancer features to include label for different stages

    Returns:
    Dataframe_cancer_with_types: A dataframe with patient IDs, cancer features, and label for cancer stage.
    """
    # -------------------T0 Vs Ta Vs Tis Vs T1 Vs T2 Vs T3 Vs T4----------------------
    csv_path = '../data/original/Al-Bladder Cancer/Data_CT only with anonymized ID 11-13-24_clean.csv'  # csv with cancer types
    df_cancer_types = pd.read_csv(csv_path)

    Dataframe_cancer_with_types = pd.merge(
        cancer_features_df, df_cancer_types[["Final Path", "Anonymized ID"]], left_on='patient_id',
        right_on="Anonymized ID", how='left'
        )

    Dataframe_cancer_with_types = Dataframe_cancer_with_types.rename(
        columns={"Final Path": "cancer_type"}
        )
    Dataframe_cancer_with_types = Dataframe_cancer_with_types.drop("Anonymized ID", axis=1)
    Dataframe_cancer_with_types = Dataframe_cancer_with_types.set_index("patient_id")
    Dataframe_cancer_with_types["cancer_sub_type_label"] = Dataframe_cancer_with_types["cancer_type"].map(
        {"T0": 0, "Ta": 1, "Tis": 2, "T1": 3, "T2": 4, "T3": 5, "T4": 6}
        ).astype(int)
    Dataframe_cancer_with_types.to_csv(f"./features/glcm_cancer_{selected_feature}_features_{max_no_of_rois}_rois_with_sub_types.csv")
    return Dataframe_cancer_with_types


def get_all_features():
    """
    Merge cancer features and healthy features

    Returns:
    Dataframe_cancer_with_types: A dataframe with patient IDs, cancer and healthy features with labels
    """
    # -------------------Cancer Vs Non-cancer-----------------------------------------
    full_features_dataframe = pd.concat([cancer_features_df, healthy_features_df], axis=0)

    csv_path = '../data/original/Al-Bladder Cancer/Data_CT only with anonymized ID 11-13-24_clean.csv'  # csv with cancer types
    df_cancer_types = pd.read_csv(csv_path)

    full_features_dataframe = pd.merge(
        full_features_dataframe, df_cancer_types[["Final Path", "Anonymized ID"]], left_on='patient_id',
        right_on="Anonymized ID", how='left'
        )
    full_features_dataframe = full_features_dataframe.rename(
        columns={"Final Path": "cancer_type"}
        )
    full_features_dataframe = full_features_dataframe.drop("Anonymized ID", axis=1)
    full_features_dataframe = full_features_dataframe.set_index("patient_id")
    full_features_dataframe.to_csv(f"./features/glcm_all_{selected_feature}_features_{max_no_of_rois}_rois.csv")
    return full_features_dataframe
