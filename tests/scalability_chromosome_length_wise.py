import random
import time
import numpy as np
from gentl.gentl import gentl
import matplotlib.pyplot as plt
# from gentl import _ga_step2_parent_selection_by_fitness_evaluation_
from gentl._ga_step2_parent_selection_by_fitness_evaluation_ import euclidean_distance

def scalability_chromosome_length_wise():
    """
    Test the scalability of the genetic algorithm by varying the length of the chromosome.
    """
    g = [0, 1]  # Gene pool
    Np_cap = 20  # Population size
    alpha = 0.05  # Mutation rate
    fitness_threshold = 0.5  # Stopping criterion for fitness

    lengths = [10, 20, 30, 40, 50, 60, 70]  # Different chromosome lengths to test
    t_len_dict = dict(zip(lengths, [[] for _ in lengths]))
    iters_len_dict = dict(zip(lengths, [[] for _ in lengths]))
    no_of_iters = 20

    for length in lengths:
        for iter in range(no_of_iters):
            goal = [random.choice(g) for _ in range(length)]  # Random goal sequence of given length
            # Adjust max generations based on chromosome length
            max_generations = 100 + length * 5  # Increase generations for longer chromosomes
            start = time.time()
            population, generation, _ = gentl(Np_cap, alpha, goal, g, max_generations, copy=True,
                                           fitness_threshold=fitness_threshold)
            end = time.time()
            time_taken = end - start
            if not population:
                continue
            t_len_dict[length].append(time_taken)
            iters_len_dict[length].append(generation)
    return t_len_dict, iters_len_dict

# Run the test
if __name__ == "__main__":
    t_len_dict, iters_len_dict = scalability_chromosome_length_wise()

    # Boxplot for iterations across different chromosome lengths
    plt.figure(figsize=(10, 6))
    plt.boxplot([iters_len_dict[length] for length in sorted(iters_len_dict.keys())],
                labels=sorted(iters_len_dict.keys()))
    plt.xlabel('Chromosome Length')
    plt.ylabel('Number of Iterations to Converge')
    plt.title('Iterations vs Chromosome Length (Boxplot)')
    plt.grid(True)
    # plt.savefig('scalability_chromosome_length_iterations_boxplot.pdf')
    plt.show()

    # Plot for time taken across different chromosome lengths
    plt.figure(figsize=(10, 6))
    plt.boxplot([t_len_dict[length] for length in sorted(t_len_dict.keys())],
                labels=sorted(t_len_dict.keys()))
    plt.xlabel('Chromosome Length')
    plt.ylabel('Time Taken (s)')
    plt.title('Time Taken vs Chromosome Length (Boxplot)')
    plt.grid(True)
    # plt.savefig('scalability_chromosome_length_time_taken_boxplot.pdf')
    plt.show()
