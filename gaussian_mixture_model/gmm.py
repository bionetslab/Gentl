import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

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

        # Divide the healthy region data into two regions
        healthy_region1_values = healthy_values[:20]
        healthy_region2_values = healthy_values[20:]

        # Convert healthy region 1 feature values to 0 or 1 using cancer region GMM means
        healthy_region1_labels = []
        for value in healthy_region1_values.flatten():
            distance_to_mean0 = abs(value - means[0])
            distance_to_mean1 = abs(value - means[1])
            label = 0 if distance_to_mean0 < distance_to_mean1 else 1
            healthy_region1_labels.append(label)

        # Convert healthy region 2 feature values to 0 or 1 using cancer region GMM means
        healthy_region2_labels = []
        for value in healthy_region2_values.flatten():
            distance_to_mean0 = abs(value - means[0])
            distance_to_mean1 = abs(value - means[1])
            label = 0 if distance_to_mean0 < distance_to_mean1 else 1
            healthy_region2_labels.append(label)

        # Store results for the current patient
        patient_results[patient_id] = {
            f'cancer_{variable_name}': cancer_values.flatten(),
            f'cancer_{variable_name}_labels': cancer_labels,
            f'healthy_region1_{variable_name}': healthy_region1_values.flatten(),
            f'healthy_region1_{variable_name}_labels': healthy_region1_labels,
            f'healthy_region2_{variable_name}': healthy_region2_values.flatten(),
            f'healthy_region2_{variable_name}_labels': healthy_region2_labels,
            f'cancer_{variable_name}_gmm_means': means
        }

    return patient_results


# Save results to CSV file
def save_results_to_csv(results, variable_name):
    results_list = []
    for patient_id, data in results.items():
        for i in range(20):
            results_list.append({
                'patient_id': patient_id,
                'roi_condition': 'cancer',
                variable_name: data[f'cancer_{variable_name}'][i],
                'label': data[f'cancer_{variable_name}_labels'][i]
            })
        for i in range(20):
            results_list.append({
                'patient_id': patient_id,
                'roi_condition': 'healthy_region1',
                variable_name: data[f'healthy_region1_{variable_name}'][i],
                'label': data[f'healthy_region1_{variable_name}_labels'][i]

            })
        for i in range(20):
            results_list.append({
                'patient_id': patient_id,
                'roi_condition': 'healthy_region2',
                variable_name: data[f'healthy_region2_{variable_name}'][i],
                'label': data[f'healthy_region2_{variable_name}_labels'][i]
            })
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(f'{variable_name}_gmm_results.csv', index=False)


def test_patient_gmm(patient_id, patient_results, variable_name):
    """
    Test GMM fitting for a specific patient and display results.

    Parameters:
    - patient_id: patient ID
    - patient_results: dictionary containing all patient data
    - variable_name: feature name
    """

    # Retrieve data for a specific patient
    patient_data = patient_results.get(patient_id)
    # Check if data was successfully retrieved
    if patient_data is None:
        print(f"No data found for patient {patient_id}")
        return

    # Extract feature values for the cancer area
    cancer_values = patient_data[f'cancer_{variable_name}'].reshape(-1, 1)

    # Refit GMM model internally for plotting
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(cancer_values)

    # Plot GMM fitting curve and the two Gaussian components for the cancer area
    plt.figure(figsize=(8, 6))
    # Plot histogram of cancer area feature values
    plt.hist(cancer_values, bins=10, density=True, alpha=0.6, color='g', label=f'{variable_name.capitalize()} (Cancer)')

    # Plot total GMM fit curve and individual Gaussian components
    x = np.linspace(cancer_values.min(), cancer_values.max(), 1000).reshape(-1, 1)
    logprob = gmm.score_samples(x)
    responsibilities = gmm.predict_proba(x)
    pdf = np.exp(logprob)
    pdf_individual = responsibilities * pdf[:, np.newaxis]
    plt.plot(x, pdf, '-k', label='GMM')
    plt.plot(x, pdf_individual[:, 0], '--r', label='GMM Component 1')
    plt.plot(x, pdf_individual[:, 1], '--b', label='GMM Component 2')
    plt.title(f'Patient {patient_id} {variable_name.capitalize()} GMM')
    plt.xlabel(variable_name.capitalize())
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    # Print lists of original and converted values
    print(f"\nPatient {patient_id}'s cancer area {variable_name} values:")
    print(patient_data[f'cancer_{variable_name}'])
    print(f"Cancer area {variable_name} labels:")
    print(patient_data[f'cancer_{variable_name}_labels'])

    print(f"\nHealthy area 1 {variable_name} values:")
    print(patient_data[f'healthy_region1_{variable_name}'])
    print(f"Healthy area 1 {variable_name} labels:")
    print(patient_data[f'healthy_region1_{variable_name}_labels'])

    print(f"\nHealthy area 2 {variable_name} values:")
    print(patient_data[f'healthy_region2_{variable_name}'])
    print(f"Healthy area 2 {variable_name} labels:")
    print(patient_data[f'healthy_region2_{variable_name}_labels'])

    print(f"\nPatient {patient_id}'s cancer area GMM means:")
    print(patient_data[f'cancer_{variable_name}_gmm_means'])



# Read feature value files
# dissimilarity_df = pd.read_csv('../scripts/glcm_dissimilarity_features.csv')
# correlation_df = pd.read_csv('../scripts/glcm_correlation_features.csv')
# energy_df = pd.read_csv('../scripts/glcm_energy_features.csv')
# contrast_df = pd.read_csv('../scripts/glcm_contrast_features.csv')
# homogeneity_df = pd.read_csv('../scripts/glcm_homogeneity_features.csv')

# Process each variable
# dissimilarity_results = process_variable_gmm(dissimilarity_df, 'dissimilarity')
# correlation_results = process_variable_gmm(correlation_df, 'correlation')
# energy_results = process_variable_gmm(energy_df, 'energy')
# contrast_results = process_variable_gmm(contrast_df, 'contrast')
# homogeneity_results = process_variable_gmm(homogeneity_df, 'homogeneity')

# save_results_to_csv(dissimilarity_results, 'dissimilarity')
# save_results_to_csv(correlation_results, 'correlation')
# save_results_to_csv(energy_results, 'energy')
# save_results_to_csv(contrast_results, 'contrast')
# save_results_to_csv(homogeneity_results, 'homogeneity')

# Call the test function
# test_patient_gmm('CT-174', dissimilarity_results, 'dissimilarity')
# test_patient_gmm('CT-174', contrast_results, 'contrast')
# test_patient_gmm('CT-174', correlation_results, 'correlation')
# test_patient_gmm('CT-174', energy_results, 'energy')
# test_patient_gmm('CT-174', homogeneity_results, 'homogeneity')
