import random
import time
import pandas as pd
import numpy as np
from gentl.gentl import gentl
# from gentl import _ga_step2_parent_selection_by_fitness_evaluation_
from gentl._ga_step2_parent_selection_by_fitness_evaluation_ import euclidean_distance

def scalability_chromosome_length_wise():
    """
    Test the scalability of the genetic algorithm by varying the length of the chromosome.
    """
    g = [1, 2, 3]  # Gene pool
    Np_cap = 20  # Population size
    alpha = 0.05  # Mutation rate
    fitness_threshold = 0.5  # Stopping criterion for fitness

    lengths = [10, 20, 30, 40, 50]  # Different chromosome lengths to test
    t_across_lengths=[]*len(lengths)
    iters_across_lengths=[]*len(lengths)
    t_len_dict=dict(zip(lengths, t_across_lengths))
    iters_len_dict = dict(zip(lengths, iters_across_lengths))
    no_of_iters=20

    for length in lengths:
        t_len_dict[length]=[]
        iters_len_dict[length]=[]
        for iter in range(no_of_iters):

            # random.seed(length)  # Set seed to maintain reproducibility
            goal = [random.choice(g) for _ in range(length)]  # Random goal sequence of given length
            # Adjust max generations based on chromosome length
            max_generations = 100 + length * 5  # Increase generations for longer chromosomes
            start=time.time()
            population, generation = gentl(Np_cap, alpha, goal, g, max_generations, copy=True, fitness_threshold=fitness_threshold)
            end=time.time()
            time_taken=end-start
            if not population:
                print(f"Chromosome Length {length}: Test failed - Population is empty.")
                continue
            t_len_dict[length].append(time_taken)
            iters_len_dict[length].append(generation)
            best_chromosome = population[0]
            best_distance = euclidean_distance(np.array(goal).flatten(), np.array(best_chromosome).flatten())
            if best_distance == 0.0:
                print(f"Solution found for Chromosome Length {length} in max_generation {max_generations}")
                print(f"Best chromosome: {best_chromosome}, Distance to goal: {best_distance}")
            else:
                print(f"Maximum generation limit reached ({max_generations}). Best chromosome for Length {length}: {best_chromosome}, Distance to goal: {best_distance}")
    return t_len_dict, iters_len_dict

# Run the test
if __name__ == "__main__":
    print("Running Scalability Test - Chromosome Length-wise...")
    t_len_dict, iters_len_dict=scalability_chromosome_length_wise()
    # Generate scalability plot:
    print(t_len_dict)
    # scalability_cr_len_iters_df=pd.DataFrame(iters_len_dict)
    # scalability_cr_len_iters_df.to_csv("scalability_cr_len_iters_df.csv", index=False)
