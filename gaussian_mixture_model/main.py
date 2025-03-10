import pandas as pd
from process_gmm import process_variable_gmm
from save_results import save_results_to_csv
from test_gmm import test_patient_gmm

# Read feature value files
# dissimilarity_df = pd.read_csv('../scripts/glcm_dissimilarity_features.csv')
# correlation_df = pd.read_csv('../scripts/glcm_correlation_features.csv')
# energy_df = pd.read_csv('../scripts/glcm_energy_features.csv')
# contrast_df = pd.read_csv('../scripts/glcm_contrast_features.csv')
# homogeneity_df = pd.read_csv('../scripts/glcm_homogeneity_features.csv')
dissimilarity_df = pd.read_csv('../scripts/extracted_glcm_features/20/glcm_dissimilarity_features_20_rois.csv')

# Process the variable
# dissimilarity_results = process_variable_gmm(dissimilarity_df, 'dissimilarity')
# correlation_results = process_variable_gmm(correlation_df, 'correlation')
# energy_results = process_variable_gmm(energy_df, 'energy')
# contrast_results = process_variable_gmm(contrast_df, 'contrast')
# homogeneity_results = process_variable_gmm(homogeneity_df, 'homogeneity')
dissimilarity_results = process_variable_gmm(dissimilarity_df, 'dissimilarity')

# Save results to CSV
# save_results_to_csv(dissimilarity_results, 'dissimilarity')
# save_results_to_csv(correlation_results, 'correlation')
# save_results_to_csv(energy_results, 'energy')
# save_results_to_csv(contrast_results, 'contrast')
# save_results_to_csv(homogeneity_results, 'homogeneity')

# Test a specific patient
test_patient_gmm('CT-139', dissimilarity_results, 'dissimilarity')
# test_patient_gmm('CT-009', dissimilarity_results, 'dissimilarity')