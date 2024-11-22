import pandas as pd

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
        region_keys = [key for key in data.keys() if key.startswith('healthy_region') and key.endswith(f'_{variable_name}')]
        for region_key in region_keys:
            region_number = region_key.split('_')[1].replace('region', '')
            for i in range(20):
                results_list.append({
                    'patient_id': patient_id,
                    'roi_condition': f'healthy_region{region_number}',
                    variable_name: data[region_key][i],
                    'label': data[f'healthy_region{region_number}_{variable_name}_labels'][i]
                })
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(f'{variable_name}_gmm_results.csv', index=False)


