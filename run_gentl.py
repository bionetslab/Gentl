import numpy as np
import pandas as pd
from gentl._ga_step2_parent_selection_by_fitness_evaluation_ import euclidean_distance, \
    parent_selection_fitness_evaluation
from gentl.gentl import gentl
from gaussian_mixture_model.process_gmm import process_variable_gmm

def run_gentl_for_feature(feature_name, feature_df, Np_cap=10, alpha=0.1, max_generations=None, fitness_threshold=0.1):
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
        best_individual, generation = gentl(Np_cap, alpha, goal_labels, g, max_generations, copy=True,
                                            fitness_threshold=fitness_threshold, p=initial_population)

        # Convert the best individual to a 1D array
        best_individual = np.array(best_individual).flatten()

        # Calculate the distance between the best individual and the goal
        best_distance = euclidean_distance(goal_labels, best_individual)

        # Store the results
        optimization_results.append({
            'patient_id': patient_id,
            'best_individual': best_individual,
            'generation': generation,
            'best_distance': best_distance
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


def sort_patients_by_distance(results):
    """
    Sort patients by the distance between the best individual and the goal.

    Parameters:
    - results: List of dictionaries containing optimization results

    Returns:
    - sorted_results: List of sorted dictionaries by distance to goal
    """
    return sorted(results, key=lambda x: x['best_distance'])


def average_generation_results_over_trials(feature_name, feature_df, num_trials=20, Np_cap=10, alpha=0.1, max_generations=None, fitness_threshold=0.1):
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

    for trial in range(num_trials):
        print(f"Running trial {trial + 1}/{num_trials} for feature '{feature_name}' with max_generations={max_generations}")
        trial_results = run_gentl_for_feature(feature_name, feature_df, Np_cap, alpha, max_generations, fitness_threshold)

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
    average_results_df.to_csv(f'{feature_name}_average_generation_results.csv', index=False)

    print(f"Average generation results saved to '{feature_name}_average_generation_results.csv'")


def average_distance_results_over_trials(feature_name, feature_df, num_trials=20, Np_cap=10, alpha=0.1, max_generations=None, fitness_threshold=0.1):
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

    for trial in range(num_trials):
        print(f"Running trial {trial + 1}/{num_trials} for feature '{feature_name}' with max_generations={max_generations}")
        trial_results = run_gentl_for_feature(feature_name, feature_df, Np_cap, alpha, max_generations, fitness_threshold)

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
    average_results_df.to_csv(f'{feature_name}_average_distance_results.csv', index=False)

    print(f"Average distance results saved to '{feature_name}_average_distance_results.csv'")


# Run the test of gentl and GMM integration
if __name__ == "__main__":
    # Example of running gentl with any feature
    feature_df = pd.read_csv('../Gentl/scripts/glcm_dissimilarity_features.csv')
    feature_name = 'dissimilarity'  # You can change this to 'correlation', 'energy', 'contrast', or 'homogeneity'

    # feature_df = pd.read_csv('../Gentl/scripts/glcm_correlation_features.csv')
    # feature_name = 'correlation'
    # feature_df = pd.read_csv('../Gentl/scripts/glcm_energy_features.csv')
    # feature_name = 'energy'
    # feature_df = pd.read_csv('../Gentl/scripts/glcm_contrast_features.csv')
    # feature_name = 'contrast'
    # feature_df = pd.read_csv('../Gentl/scripts/glcm_homogeneity_features.csv')
    # feature_name = 'homogeneity'

    # Sort and display patients by distance
    optimization_distance_results = run_gentl_for_feature(feature_name, feature_df, max_generations=8)
    sorted_by_distance = sort_patients_by_distance(optimization_distance_results)
    print("\nSorted results by distance between best individual and goal:")
    for result in sorted_by_distance:
        print(f"Patient {result['patient_id']}, Distance to goal: {result['best_distance']}")

    # Sort and display patients by generation
    optimization_generation_results = run_gentl_for_feature(feature_name, feature_df, max_generations=50)
    sorted_by_generation = sort_patients_by_generation(optimization_generation_results)
    print("\nSorted results by number of iterations:")
    for result in sorted_by_generation:
        print(f"Patient {result['patient_id']}, Number of iterations: {result['generation']}")

    # Run multiple trials and save average results to CSV
    # average_distance_results_over_trials(feature_name, feature_df, max_generations=8)
    # average_generation_results_over_trials(feature_name, feature_df, max_generations=50)



