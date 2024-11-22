import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

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
    # pdf_path = f'/Users/wushengyang/Desktop/Patient_{patient_id}_{variable_name}_GMM.pdf'
    # plt.savefig(pdf_path)
    plt.show()

    # Print lists of original and converted values
    print(f"\nPatient {patient_id}'s cancer area {variable_name} values:")
    print(patient_data[f'cancer_{variable_name}'])
    print(f"Cancer area {variable_name} labels:")
    print(patient_data[f'cancer_{variable_name}_labels'])

    num_regions = len(
        [key for key in patient_data.keys() if key.startswith('healthy_region') and key.endswith(variable_name)])
    for region_idx in range(num_regions):
        print(f"\nHealthy area {region_idx + 1} {variable_name} values:")
        print(patient_data[f'healthy_region{region_idx + 1}_{variable_name}'])
        print(f"Healthy area {region_idx + 1} {variable_name} labels:")
        print(patient_data[f'healthy_region{region_idx + 1}_{variable_name}_labels'])

    print(f"\nPatient {patient_id}'s cancer area GMM means:")
    print(patient_data[f'cancer_{variable_name}_gmm_means'])

