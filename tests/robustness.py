import numpy as np
import random
from gentl.gentl import gentl
# from gentl import _ga_step2_parent_selection_by_fitness_evaluation_
from gentl._ga_step2_parent_selection_by_fitness_evaluation_ import euclidean_distance

def robustness_test():
    """
    Test the robustness of the genetic algorithm by running it multiple times with different initial conditions.
    """
    goal = [1, 2, 3, 3, 2, 1, 2, 2, 1, 3]  # Target sequence
    g = [1, 2, 3]  # Gene pool
    Np_cap = 20  # Population size
    alpha = 0.05  # Mutation rate (5%)
    max_generations = 100  # Maximum number of generations
    fitness_threshold = 0.5  # Stopping criterion for fitness

    results = []

    # Run the genetic algorithm multiple times with different random seeds
    for seed in range(5):
        random.seed(seed)
        np.random.seed(seed)
        population = gentl(Np_cap, alpha, goal, g, max_generations, copy=True, fitness_threshold=fitness_threshold)
        if not population:
            print(f"Seed {seed}: Test failed - Population is empty.")
            continue
        best_chromosome = population[0]
        best_distance = euclidean_distance(goal, best_chromosome)
        results.append((best_chromosome, best_distance))
        print(f"Seed {seed}: Best chromosome: {best_chromosome}, Distance to goal: {best_distance}")

    return results


if __name__ == "__main__":
    print("Running Robustness Test...")
    robustness_test()