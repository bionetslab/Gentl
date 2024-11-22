from sklearn.mixture import GaussianMixture

def process_variable_gmm(dataframe, variable_name):
    """
    Fit GMM and convert labels for a specified variable.

    Parameters:
    - dataframe: DataFrame containing feature data
    - variable_name: the name of the variable to process (string), e.g., 'dissimilarity'

    Returns:
    - patient_results: dictionary containing results for each patient
    """
    # Get a list of unique patient IDs
    patient_ids = dataframe['patient_id'].unique()

    # Initialize dictionary to store results
    patient_results = {}

    for patient_id in patient_ids:
        # Get data for the current patient
        patient_data = dataframe[dataframe['patient_id'] == patient_id]

        # Extract data for the cancer region
        cancer_data = patient_data[patient_data['roi_condition'] == 'cancer']
        cancer_values = cancer_data[variable_name].values.reshape(-1, 1)

        # Fit a bi-modal GMM to the feature values of the cancer region
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(cancer_values)

        # Retrieve the means of the GMM model
        means = gmm.means_.flatten()

        # Sort the means for consistency
        means.sort()

        # Convert cancer region feature values to 0 or 1 based on distance to means
        cancer_labels = []
        for value in cancer_values.flatten():
            distance_to_mean0 = abs(value - means[0])
            distance_to_mean1 = abs(value - means[1])
            label = 0 if distance_to_mean0 < distance_to_mean1 else 1
            cancer_labels.append(label)

        # Extract data for the healthy (non-cancerous) region
        healthy_data = patient_data[patient_data['roi_condition'] == 'healthy']
        healthy_values = healthy_data[variable_name].values.reshape(-1, 1)

        # Divide the healthy region data into multiple regions based on the number of available values
        num_regions = len(healthy_values) // 20
        patient_results[patient_id] = {
            f'cancer_{variable_name}': cancer_values.flatten(),
            f'cancer_{variable_name}_labels': cancer_labels,
            f'cancer_{variable_name}_gmm_means': means
        }

        for region_idx in range(num_regions):
            start_idx = region_idx * 20
            end_idx = start_idx + 20
            region_values = healthy_values[start_idx:end_idx]

            # Convert healthy region feature values to 0 or 1 using cancer region GMM means
            healthy_region_labels = []
            for value in region_values.flatten():
                distance_to_mean0 = abs(value - means[0])
                distance_to_mean1 = abs(value - means[1])
                label = 0 if distance_to_mean0 < distance_to_mean1 else 1
                healthy_region_labels.append(label)

            # Store healthy region data
            patient_results[patient_id][f'healthy_region{region_idx + 1}_{variable_name}'] = region_values.flatten()
            patient_results[patient_id][f'healthy_region{region_idx + 1}_{variable_name}_labels'] = healthy_region_labels

    return patient_results

