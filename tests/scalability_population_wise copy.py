from gentl.gentl import gentl
# from gentl import _ga_step2_parent_selection_by_fitness_evaluation_
from gentl._ga_step2_parent_selection_by_fitness_evaluation_ import euclidean_distance
import time
import numpy as np
import matplotlib.pyplot as plt

def scalability_population_wise():
    """
    Test the scalability of the genetic algorithm by varying the size of the population.
    """
    goal = [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0]  # Target sequence
    g = [0, 1]  # Gene pool
    alpha = 0.05  # Mutation rate
    max_generations = 100  # Maximum number of generations
    fitness_threshold = 0.5  # Stopping criterion for fitness

    population_sizes = [20, 40, 60, 80, 100]  # Different population sizes to test
    t_pop_dict = dict(zip(population_sizes, [[] for _ in population_sizes]))
    iters_pop_dict = dict(zip(population_sizes, [[] for _ in population_sizes]))
    no_of_iters = 20

    for Np_cap in population_sizes:
        for iter in range(no_of_iters):
            start = time.time()
            population, generation, _ = gentl(Np_cap, alpha, goal, g, max_generations, copy=True,
                                       fitness_threshold=fitness_threshold)
            end = time.time()
            time_taken = end - start
            if not population:
                continue
            t_pop_dict[Np_cap].append(time_taken)
            iters_pop_dict[Np_cap].append(generation)

    return t_pop_dict, iters_pop_dict

# Run the test
if __name__ == "__main__":
    t_pop_dict, iters_pop_dict = scalability_population_wise()

    # Boxplot for iterations across different population sizes
    plt.figure(figsize=(10, 6))
    plt.boxplot([iters_pop_dict[size] for size in sorted(iters_pop_dict.keys())], labels=sorted(iters_pop_dict.keys()))
    plt.xlabel('Population Size')
    plt.ylabel('Number of Iterations to Converge')
    plt.title('Iterations vs Population Size (Boxplot)')
    plt.grid(True)
    # plt.savefig('scalability_population_size_iterations_boxplot.pdf')
    plt.show()

    # Plot for time taken across different chromosome lengths
    plt.figure(figsize=(10, 6))
    plt.boxplot([t_pop_dict[length] for length in sorted(t_pop_dict.keys())],
                labels=sorted(t_pop_dict.keys()))
    plt.xlabel('Population Size')
    plt.ylabel('Time Taken (s)')
    plt.title('Time Taken vs Population Size (Boxplot)')
    plt.grid(True)
    # plt.savefig('scalability_population_size_time_taken_boxplot.pdf')
    plt.show()
