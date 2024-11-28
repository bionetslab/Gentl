import time
import numpy as np
import matplotlib.pyplot as plt
from gentl.gentl import gentl
# from gentl import _ga_step2_parent_selection_by_fitness_evaluation_
from gentl._ga_step2_parent_selection_by_fitness_evaluation_ import euclidean_distance

def scalability_iteration_wise():
    """
    Test the scalability of the genetic algorithm by varying the number of iterations allowed.
    """
    goal = [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0]  # Target sequence
    g = [0, 1]  # Gene pool
    Np_cap = 40  # Population size
    alpha = 0.05  # Mutation rate
    fitness_threshold = 0.5  # Stopping criterion for fitness

    max_generations_list = [35, 40, 45, 50, 55, 60, 65, 70, 75, 80]  # Different maximum generations to test
    t_gen_dict = dict(zip(max_generations_list, [[] for _ in max_generations_list]))
    iters_gen_dict = dict(zip(max_generations_list, [[] for _ in max_generations_list]))
    no_of_iters = 20

    for max_generations in max_generations_list:
        for iter in range(no_of_iters):
            start = time.time()
            population, generation, _ = gentl(Np_cap, alpha, goal, g, max_generations, copy=True,
                                           fitness_threshold=fitness_threshold)
            end = time.time()
            time_taken = end - start
            if not population:
                continue
            t_gen_dict[max_generations].append(time_taken)
            iters_gen_dict[max_generations].append(generation)
    return t_gen_dict, iters_gen_dict

# Run the test
if __name__ == "__main__":
    t_gen_dict, iters_gen_dict = scalability_iteration_wise()

    # Boxplot for iterations across different max generations
    plt.figure(figsize=(10, 6))
    plt.boxplot([iters_gen_dict[gen] for gen in sorted(iters_gen_dict.keys())], labels=sorted(iters_gen_dict.keys()))
    plt.xlabel('Max Generations')
    plt.ylabel('Number of Iterations to Converge')
    plt.title('Iterations vs Max Generations (Boxplot)')
    plt.grid(True)
    # plt.savefig('scalability_max_generations_iterations_boxplot.pdf')
    plt.show()

    # Plot for time taken across different chromosome lengths
    plt.figure(figsize=(10, 6))
    plt.boxplot([t_gen_dict[length] for length in sorted(t_gen_dict.keys())],
                labels=sorted(t_gen_dict.keys()))
    plt.xlabel('Max Generations')
    plt.ylabel('Time Taken (s)')
    plt.title('Time Taken vs Max Generations (Boxplot)')
    plt.grid(True)
    # plt.savefig('scalability_max_generations_time_taken_boxplot.pdf')
    plt.show()
