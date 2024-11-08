import random
from gentl.gentl import gentl
# from gentl import _ga_step2_parent_selection_by_fitness_evaluation_
from gentl._ga_step2_parent_selection_by_fitness_evaluation_ import euclidean_distance

def scalability_iteration_wise():
    """
    Test the scalability of the genetic algorithm by varying the number of iterations allowed.
    """
    goal = [1, 2, 3, 3, 2, 1, 2, 2, 1, 3]  # Target sequence
    g = [1, 2, 3]  # Gene pool
    Np_cap = 20  # Population size
    alpha = 0.05  # Mutation rate
    fitness_threshold = 0.5  # Stopping criterion for fitness

    max_generations_list = [10, 50, 100, 200, 500]  # Different maximum generations to test

    for max_generations in max_generations_list:
        random.seed(max_generations)  # Set seed to maintain reproducibility
        population = gentl(Np_cap, alpha, goal, g, max_generations, copy=True, fitness_threshold=fitness_threshold)
        if not population:
            print(f"Max Generations {max_generations}: Test failed - Population is empty.")
            continue
        best_chromosome = population[0]
        best_distance = euclidean_distance(goal, best_chromosome)
        print(f"Max Generations {max_generations}: Best chromosome: {best_chromosome}, Distance to goal: {best_distance}")


# Run the test
if __name__ == "__main__":
    print("Running Scalability Test - Iteration-wise...")
    scalability_iteration_wise()
