import numpy as np
import pandas as pd
from gentl._ga_step2_parent_selection_by_fitness_evaluation_ import euclidean_distance, absolute_distance, parent_selection_fitness_evaluation
from gentl.gentl import gentl
from gaussian_mixture_model.process_gmm import process_variable_gmm

def run_gentl_for_feature(feature_name, feature_df, Np_cap=None, alpha=0.1, max_generations=None, fitness_threshold=0.1, distance_mode='euclidean'):
    """
    Run genetic algorithm for a specific feature and store the results.

    Parameters:
    - feature_name: Name of the feature to process (e.g., 'dissimilarity')
    - feature_df: DataFrame containing feature data
    - Np_cap: Population size for the genetic algorithm
    - alpha: Mutation rate for the genetic algorithm
    - max_generations: Maximum number of iterations for the genetic algorithm
    - fitness_threshold: Distance threshold for stopping the genetic algorithm
    """
    # Process GMM for the specified feature
    feature_results = process_variable_gmm(feature_df, feature_name)

    # Initialize a results list to store outcomes for all patients
    optimization_results = []

    # Select the appropriate distance function based on the mode
    if distance_mode == 'euclidean':
        distance_func = euclidean_distance
    elif distance_mode == 'absolute':
        distance_func = absolute_distance
    else:
        raise ValueError("Invalid distance mode. Choose 'euclidean' or 'absolute'.")

    # Iterate over each patient and optimize using the genetic algorithm
    for patient_id, patient_data in feature_results.items():
        goal_labels = patient_data[f'cancer_{feature_name}_labels']
        healthy_labels = []

        # Gather labels from all healthy regions
        num_regions = len(
            [key for key in patient_data.keys() if key.startswith('healthy_region') and key.endswith(feature_name)])
        for region_idx in range(1, num_regions + 1):
            healthy_labels.append(patient_data[f'healthy_region{region_idx}_{feature_name}_labels'])

        # Define the initial population as label sets from all non-cancer regions
        initial_population = healthy_labels

        # Define gene pool (0 and 1)
        g = [0, 1]

        # Run the genetic algorithm
        best_individual, generation, mean_distances = gentl(Np_cap, alpha, goal_labels, g, max_generations, copy=True,
                                            fitness_threshold=fitness_threshold, mu=1, p=initial_population, distance_mode=distance_mode)

        # Convert the best individual to a 1D array
        best_individual = np.array(best_individual).flatten()

        # Calculate the distance between the best individual and the goal
        best_distance = distance_func(goal_labels, best_individual)

        # Store the results
        optimization_results.append({
            'patient_id': patient_id,
            'best_individual': best_individual,
            'generation': generation,
            'best_distance': best_distance,
            'mean_distances': mean_distances
        })

        # Output optimization results
        # print(f"Patient {patient_id}'s optimization results:")
        # print(f"Best individual: {best_individual}")
        # print(f"Number of iterations: {generation}")
        # print(f"Distance between best individual and goal: {best_distance}\n")

    return optimization_results


def sort_patients_by_generation(results):
    """
    Sort patients by the number of iterations required.

    Parameters:
    - results: List of dictionaries containing optimization results

    Returns:
    - sorted_results: List of sorted dictionaries by number of iterations
    """
    return sorted(results, key=lambda x: x['generation'])


def sort_patients_by_best_distance(results):
    """
    Sort patients by the distance between the best individual and the goal.

    Parameters:
    - results: List of dictionaries containing optimization results

    Returns:
    - sorted_results: List of sorted dictionaries by distance to goal
    """
    return sorted(results, key=lambda x: x['best_distance'])

def average_generation_results_over_trials(feature_name, feature_df, num_trials=20, Np_cap=10, alpha=0.1, max_generations=None, fitness_threshold=0.1, distance_mode='euclidean'):
    """
    Run the genetic algorithm multiple times and save the average generation results to a CSV file.

    Parameters:
    - feature_name: Name of the feature to process (e.g., 'dissimilarity')
    - feature_df: DataFrame containing feature data
    - num_trials: Number of times to run the genetic algorithm
    - Np_cap: Population size for the genetic algorithm
    - alpha: Mutation rate for the genetic algorithm
    - max_generations: Maximum number of iterations for the genetic algorithm
    - fitness_threshold: Distance threshold for stopping the genetic algorithm
    """
    cumulative_results = {}

    # Determine the total number of healthy regions
    patient_data_example = list(process_variable_gmm(feature_df, feature_name).values())[0]
    total_healthy_region_number = len(
        [key for key in patient_data_example.keys() if key.startswith('healthy_region') and key.endswith(feature_name)])

    for trial in range(num_trials):
        print(f"Running trial {trial + 1}/{num_trials} for feature '{feature_name}' with max_generations={max_generations} using {distance_mode}")
        trial_results = run_gentl_for_feature(feature_name, feature_df, Np_cap, alpha, max_generations, fitness_threshold, distance_mode=distance_mode)

        for result in trial_results:
            patient_id = result['patient_id']
            if patient_id not in cumulative_results:
                cumulative_results[patient_id] = {'generation': []}
            cumulative_results[patient_id]['generation'].append(result['generation'])

    # Calculate the average results and save to CSV
    average_results_list = []
    for patient_id, metrics in cumulative_results.items():
        avg_generation = np.mean(metrics['generation'])
        average_results_list.append({'patient_id': patient_id, 'average_generation': avg_generation})

    average_results_list = sorted(average_results_list, key=lambda x: x['average_generation'])
    average_results_df = pd.DataFrame(average_results_list)
    average_results_df.to_csv(f'{feature_name}_average_generation_{distance_mode}_distance_results_{total_healthy_region_number}_rois.csv', index=False)

    print(f"Average generation results saved to '{feature_name}_average_generation_{distance_mode}_distance_results_{total_healthy_region_number}_rois.csv'")


def average_best_distance_results_over_trials(feature_name, feature_df, num_trials=20, Np_cap=10, alpha=0.1, max_generations=None, fitness_threshold=0.1, distance_mode='euclidean'):
    """
    Run the genetic algorithm multiple times and save the average distance results to a CSV file.

    Parameters:
    - feature_name: Name of the feature to process (e.g., 'dissimilarity')
    - feature_df: DataFrame containing feature data
    - num_trials: Number of times to run the genetic algorithm
    - Np_cap: Population size for the genetic algorithm
    - alpha: Mutation rate for the genetic algorithm
    - max_generations: Maximum number of iterations for the genetic algorithm
    - fitness_threshold: Distance threshold for stopping the genetic algorithm
    """
    cumulative_results = {}

    # Determine the total number of healthy regions
    patient_data_example = list(process_variable_gmm(feature_df, feature_name).values())[0]
    total_healthy_region_number = len(
        [key for key in patient_data_example.keys() if key.startswith('healthy_region') and key.endswith(feature_name)])

    for trial in range(num_trials):
        print(f"Running trial {trial + 1}/{num_trials} for feature '{feature_name}' with max_generations={max_generations} using {distance_mode}")
        trial_results = run_gentl_for_feature(feature_name, feature_df, Np_cap, alpha, max_generations, fitness_threshold, distance_mode=distance_mode)

        for result in trial_results:
            patient_id = result['patient_id']
            if patient_id not in cumulative_results:
                cumulative_results[patient_id] = {'best_distance': []}
            cumulative_results[patient_id]['best_distance'].append(result['best_distance'])

    # Calculate the average results and save to CSV
    average_results_list = []
    for patient_id, metrics in cumulative_results.items():
        avg_distance = np.mean(metrics['best_distance'])
        average_results_list.append({'patient_id': patient_id, 'average_distance': avg_distance})

    average_results_list = sorted(average_results_list, key=lambda x: x['average_distance'])
    average_results_df = pd.DataFrame(average_results_list)
    average_results_df.to_csv(f'{feature_name}_average_best_{distance_mode}_distance_results_{total_healthy_region_number}_rois.csv', index=False)

    print(f"Average distance results saved to '{feature_name}_average_best_{distance_mode}_distance_results_{total_healthy_region_number}_rois.csv'")


def average_mean_distance_results_over_trials(feature_name, feature_df, num_trials=20, Np_cap=None, alpha=0.1,
                                             max_generations=None, fitness_threshold=0.1, distance_mode='euclidean'):
    """
    Run the genetic algorithm multiple times and save the average distance results to a CSV file.

    Parameters:
    - feature_name: Name of the feature to process (e.g., 'dissimilarity')
    - feature_df: DataFrame containing feature data
    - num_trials: Number of times to run the genetic algorithm
    - Np_cap: Population size for the genetic algorithm
    - alpha: Mutation rate for the genetic algorithm
    - max_generations: Maximum number of iterations for the genetic algorithm
    - fitness_threshold: Distance threshold for stopping the genetic algorithm
    """
    cumulative_results = {}

    # Determine the total number of healthy regions
    patient_data_example = list(process_variable_gmm(feature_df, feature_name).values())[0]
    total_healthy_region_number = len(
        [key for key in patient_data_example.keys() if key.startswith('healthy_region') and key.endswith(feature_name)])

    for trial in range(num_trials):
        print(f"Running trial {trial + 1}/{num_trials} for feature '{feature_name}' with max_generations={max_generations} using {distance_mode}")
        trial_results = run_gentl_for_feature(feature_name, feature_df, Np_cap, alpha, max_generations, fitness_threshold, distance_mode=distance_mode)

        for result in trial_results:
            patient_id = result['patient_id']
            mean_distances = result['mean_distances']
            selected_distance = mean_distances[-1]  # last generation of mean distance

            if patient_id not in cumulative_results:
                cumulative_results[patient_id] = {'selected_distances': []}
            cumulative_results[patient_id]['selected_distances'].append(selected_distance)

    # Calculate the average results and save to CSV
    average_results_list = []
    for patient_id, metrics in cumulative_results.items():
        avg_distance = np.mean(metrics['selected_distances'])
        average_results_list.append({'patient_id': patient_id, 'average_distance': avg_distance})

    average_results_list = sorted(average_results_list, key=lambda x: x['average_distance'])
    average_results_df = pd.DataFrame(average_results_list)
    average_results_df.to_csv(f'{feature_name}_average_mean_{distance_mode}_distance_results_{total_healthy_region_number}_rois.csv',
                              index=False)

    print(f"Average mean_distance results saved to '{feature_name}_average_mean_{distance_mode}_distance_results_{total_healthy_region_number}_rois.csv'")


# Run the test of gentl and GMM integration
if __name__ == "__main__":
    # Example of running gentl with any feature
    feature_df = pd.read_csv('../Gentl/glcm_extracted_features_results/glcm_dissimilarity_features_20_rois.csv')
    feature_name = 'dissimilarity'  # You can change this to 'correlation', 'energy', 'contrast', or 'homogeneity'
    # feature_df = pd.read_csv('../Gentl/scripts/glcm_correlation_features_100_rois.csv')
    # feature_name = 'correlation'
    # feature_df = pd.read_csv('../Gentl/scripts/glcm_energy_features_100_rois.csv')
    # feature_name = 'energy'
    # feature_df = pd.read_csv('../Gentl/scripts/glcm_contrast_features_100_rois.csv')
    # feature_name = 'contrast'
    # feature_df = pd.read_csv('../Gentl/scripts/glcm_homogeneity_features_100_rois.csv')
    # feature_name = 'homogeneity'

    # Case1.1: Sort and display patients by best distance
    # optimization_distance_results = run_gentl_for_feature(feature_name, feature_df, Np_cap=57, alpha=0.05, max_generations=8)
    # sorted_by_distance = sort_patients_by_best_distance(optimization_distance_results)
    # print("\nSorted results by best distance between best individual and goal:")
    # for result in sorted_by_distance:
    #     print(f"Patient {result['patient_id']}, Best distance to goal: {result['best_distance']}")

    # Case1.2: Sort and display patients by mean distance
    # optimization_distance_results = run_gentl_for_feature(feature_name, feature_df,  Np_cap=57, alpha=0.05, max_generations=4, distance_mode='absolute')
    # sorted_results = sorted(optimization_distance_results, key=lambda x: x['mean_distances'][-1])
    # print("\nSorted results by mean distance between best individual and goal:")
    # for result in sorted_results:
    #     mean_distances = result['mean_distances']
    #     print(f"Patient {result['patient_id']}, Distance to goal: {mean_distances[-1]}")

    # Case2: Sort and display patients by generation:
    # hint: set different Np_cap values for different number of rois: Np_cap≥rois
    # 10rois->15np, 20rois->25np, 50rois->55np, 100rois->110np
    # optimization_generation_results = run_gentl_for_feature(feature_name, feature_df, Np_cap=50, alpha=0.05, max_generations=10000)
    # sorted_by_generation = sort_patients_by_generation(optimization_generation_results)
    # print("\nSorted results by number of iterations:")
    # for result in sorted_by_generation:
    #     print(f"Patient {result['patient_id']}, Number of iterations: {result['generation']}")
    #
    # Run multiple trials and save average results to CSV
    # average_best_distance_results_over_trials(feature_name, feature_df, max_generations=4)
    # average_mean_distance_results_over_trials(feature_name, feature_df, num_trials=20, Np_cap=12, alpha=0.05,
    #                                           max_generations=8, distance_mode='absolute')
    average_generation_results_over_trials(feature_name, feature_df, num_trials=20, Np_cap=20, alpha=0.05,
                                           max_generations=200)




