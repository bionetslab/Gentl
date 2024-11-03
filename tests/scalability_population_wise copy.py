from gentl.gentl import gentl
# from gentl import _ga_step2_parent_selection_by_fitness_evaluation_
from gentl._ga_step2_parent_selection_by_fitness_evaluation_ import euclidean_distance

def test_genetic_algorithm():
    # Define goal sequence (e.g., a simple string sequence for testing)
    goal = [1, 2, 3, 3, 2, 1, 2 ,2, 1, 3]  # A goal sequence for simplicity

    # Define gene pool (list of permissible values for each gene)
    g = [1, 2, 3]

    # Set genetic algorithm parameters
    Np_cap = 10           # Population size
    alpha = 0.05          # Mutation rate (5%)
    max_generations = 100  # Maximum number of generations
    fitness_threshold = 0.5  # Maximum allowable distance from goal

    # Run the genetic algorithm
    population = gentl(Np_cap, alpha, goal, g, max_generations, copy=True, fitness_threshold=fitness_threshold)
    # Check if the population is empty
    if not population:
        print("Test failed: Population is empty.")
        return
    # Print the result
    best_chromosome = population[0]
    best_distance = euclidean_distance(goal, best_chromosome)
    print(f"Best chromosome: {best_chromosome}")
    print(f"Distance to goal: {best_distance}")

# Run the test
if __name__ == "__main__":
    test_genetic_algorithm()