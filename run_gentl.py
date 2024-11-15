import numpy as np
import pandas as pd
from gentl._ga_step2_parent_selection_by_fitness_evaluation_ import euclidean_distance, parent_selection_fitness_evaluation
from gentl.gentl import gentl
from gaussian_mixture_model.gmm import process_variable_gmm

def run_gentl_with_gmm_results():
    """
    Run genetic algorithm with previous GMM results and observe the optimization effects.
    """

    dissimilarity_df = pd.read_csv('../Gentl/scripts/glcm_dissimilarity_features.csv')
    dissimilarity_results = process_variable_gmm(dissimilarity_df, 'dissimilarity')

    # Initialize a results list to store outcomes for all patients
    optimization_results = []

    # Iterate over each patient and optimize using the genetic algorithm
    for patient_id, patient_data in dissimilarity_results.items():
        goal_labels = patient_data['cancer_dissimilarity_labels']
        healthy_region1_labels = patient_data['healthy_region1_dissimilarity_labels']
        healthy_region2_labels = patient_data['healthy_region2_dissimilarity_labels']

        # Define the initial population as label sets from two non-cancer regions
        initial_population = [healthy_region1_labels, healthy_region2_labels]

        # Define gene pool (0 and 1)
        g = [0, 1]

        # Set genetic algorithm parameters
        Np_cap = 10  # Population size
        alpha = 0.1  # Mutation rate (10%)
        max_generations = 50  # Maximum number of iterations
        fitness_threshold = 0.1  # Distance threshold

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
        print(f"\nPatient {patient_id}'s optimization results:")
        print(f"Best individual: {best_individual}")
        print(f"Number of iterations: {generation}")
        print(f"Distance between best individual and goal: {best_distance}")


    # Sort by number of iterations and output
    sorted_by_generation = sorted(optimization_results, key=lambda x: x['generation'])
    print("\nSorted results by number of iterations:")
    for result in sorted_by_generation:
        print(f"Patient {result['patient_id']}, Number of iterations: {result['generation']}")

    # Optionally, sort by distance between best individual and goal and output
    # sorted_by_distance = sorted(optimization_results, key=lambda x: x['best_distance'])
    # print("\nSorted results by distance between best individual and goal:")
    # for result in sorted_by_distance:
    #     print(f"Patient {result['patient_id']}, Distance between best individual and goal: {result['best_distance']}")


# Run the test of gentl and GMM integration
if __name__ == "__main__":
    run_gentl_with_gmm_results()
