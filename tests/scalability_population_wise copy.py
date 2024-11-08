import random
from gentl.gentl import gentl
# from gentl import _ga_step2_parent_selection_by_fitness_evaluation_
from gentl._ga_step2_parent_selection_by_fitness_evaluation_ import euclidean_distance

def scalability_population_wise():
    """
    Test the scalability of the genetic algorithm by varying the size of the population.
    """
    goal = [1, 2, 3, 3, 2, 1, 2, 2, 1, 3]  # Target sequence
    g = [1, 2, 3]  # Gene pool
    alpha = 0.05  # Mutation rate
    max_generations = 100  # Maximum number of generations
    fitness_threshold = 0.5  # Stopping criterion for fitness

    population_sizes = [10, 50, 100, 200, 500]  # Different population sizes to test

    for Np_cap in population_sizes:
        random.seed(Np_cap)  # Set seed to maintain reproducibility
        population = gentl(Np_cap, alpha, goal, g, max_generations, copy=True, fitness_threshold=fitness_threshold)
        if not population:
            print(f"Population Size {Np_cap}: Test failed - Population is empty.")
            continue
        best_chromosome = population[0]
        best_distance = euclidean_distance(goal, best_chromosome)
        print(f"Population Size {Np_cap}: Best chromosome: {best_chromosome}, Distance to goal: {best_distance}")


# Run the test
if __name__ == "__main__":
    print("Running Scalability Test - Population Size-wise...")
    scalability_population_wise()