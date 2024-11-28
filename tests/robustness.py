import random
import time
import numpy as np
import matplotlib.pyplot as plt
from gentl.gentl import gentl
from gentl._ga_step2_parent_selection_by_fitness_evaluation_ import euclidean_distance

def test_random_initializations():
    """
    Test 1: Multiple Random Initializations with Fixed Seed
    """
    goal = [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0]  # Target sequence
    g = [0, 1]  # Gene pool
    Np_cap = 20  # Population size cap
    max_generations = 200  # Maximum number of generations
    fitness_threshold = 0.5  # Stopping criterion for fitness
    no_of_runs = 20
    seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # Different seeds for reproducibility

    all_convergence_generations = []
    all_convergence_times = []
    for seed in seeds:
        convergence_generations = []
        convergence_times = []
        random.seed(seed)
        for run in range(no_of_runs):
            start = time.time()
            population, generation, _ = gentl(Np_cap, 0.05, goal, g, max_generations, copy=True, fitness_threshold=fitness_threshold)
            end = time.time()
            time_taken = end - start

            if population:
                convergence_generations.append(generation)
                convergence_times.append(time_taken)

        all_convergence_generations.append(convergence_generations)
        all_convergence_times.append(convergence_times)

    # Boxplot for convergence generations and times across different seeds
    plt.figure(figsize=(12, 6))
    plt.boxplot(all_convergence_generations, labels=[f'Seed {seed}' for seed in seeds])
    plt.xlabel('Seeds')
    plt.ylabel('Generations to Converge')
    plt.title('Convergence Generations for Different Seeds')
    plt.grid(True)
    # plt.savefig('robustness_convergence_generations_seeds.pdf')
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.boxplot(all_convergence_times, labels=[f'Seed {seed}' for seed in seeds])
    plt.xlabel('Seeds')
    plt.ylabel('Time Taken (s)')
    plt.title('Convergence Time for Different Seeds')
    plt.grid(True)
    # plt.savefig('robustness_convergence_times_seeds.pdf')
    plt.show()

def test_extreme_mutation_rates():
    """
    Test 2: Varying Extreme Mutation Rates (Only Iteration Count)
    """
    goal = [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0]  # Target sequence
    g = [0, 1]  # Gene pool
    Np_cap = 20  # Population size cap
    max_generations = 200  # Maximum number of generations
    fitness_threshold = 0.5  # Stopping criterion for fitness
    alpha_values = [0.001, 0.05, 0.1, 0.15, 0.9]  # Extreme mutation rates
    no_of_runs = 20

    convergence_generations_extreme = []
    for alpha in alpha_values:
        generations = []
        for run in range(no_of_runs):
            population, generation, _ = gentl(Np_cap, alpha, goal, g, max_generations, copy=True, fitness_threshold=fitness_threshold)
            if population:
                generations.append(generation)
        convergence_generations_extreme.append(generations)

    # Boxplot for convergence generations with extreme mutation rates
    plt.figure(figsize=(10, 6))
    plt.boxplot(convergence_generations_extreme, labels=[str(alpha) for alpha in alpha_values])
    plt.xlabel('Mutation Rate')
    plt.ylabel('Generations to Converge')
    plt.title('Convergence Generations with different Mutation Rates')
    plt.grid(True)
    # plt.savefig('robustness_different_mutations_generations.pdf')
    plt.show()

def test_local_optima():
    """
    Test 3: Large Initial Population vs Np_cap (Visualization of Local Optima)
    """
    goal = [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0]  # Target sequence
    g = [0, 1]  # Gene pool
    initial_size = 50
    max_generations = 100  # Maximum number of generations
    fitness_threshold = 0.1  # Stopping criterion for fitness
    Np_cap_values = [25, 26, 28, 30, 40, 50, 60]  # Different Np_cap values to test

    plt.figure(figsize=(12, 8))
    threshold_line = [fitness_threshold] * max_generations

    for Np_cap in Np_cap_values:
        initial_population = [random.choices(g, k=len(goal)) for _ in range(initial_size)]
        population, generation, mean_distances = gentl(Np_cap, 0.05, goal, g, max_generations, copy=True,
                                                       fitness_threshold=fitness_threshold, mu=1, p=initial_population)
        if mean_distances:
            plt.plot(range(len(mean_distances)), mean_distances, marker='o', linestyle='-', alpha=0.7,
                     label=f'Np_cap = {Np_cap}')

    plt.plot(range(max_generations), threshold_line, linestyle='--', color='red', label='Threshold Distance')
    plt.xlabel('Generation')
    plt.ylabel('Distance to Goal')
    plt.title('Distance to Goal per Generation for Different Np_cap Values')
    plt.legend()
    plt.grid(True)
    # plt.savefig('robustness_different_Np_caps_distance.pdf')
    plt.show()

def test_multiple_Np_caps_boxplots():
    """
    Test: Multiple Runs for Different Np_cap Values (Boxplots for Final Iterations and Time Taken)
    """
    goal = [0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0]  # Target sequence
    g = [0, 1]  # Gene pool
    initial_size = 50
    max_generations = 200  # Maximum number of generations
    fitness_threshold = 0.1  # Stopping criterion for fitness
    Np_cap_values = [28, 30, 40, 50, 60]  # Different Np_cap values to test
    no_of_runs = 10

    convergence_generations = []
    convergence_times = []

    for Np_cap in Np_cap_values:
        generations_per_cap = []
        times_per_cap = []
        for run in range(no_of_runs):
            initial_population = [random.choices(g, k=len(goal)) for _ in range(initial_size)]
            start_time = time.time()
            population, generation, _ = gentl(Np_cap, 0.05, goal, g, max_generations, copy=True,
                                             fitness_threshold=fitness_threshold, mu=1, p=initial_population)
            end_time = time.time()
            time_taken = end_time - start_time

            if population:
                generations_per_cap.append(generation)
                times_per_cap.append(time_taken)

        convergence_generations.append(generations_per_cap)
        convergence_times.append(times_per_cap)

    # Boxplot for convergence generations for different Np_cap values
    plt.figure(figsize=(12, 6))
    plt.boxplot(convergence_generations, labels=[f'Np_cap = {Np_cap}' for Np_cap in Np_cap_values])
    plt.xlabel('Np_cap Values')
    plt.ylabel('Generations to Converge')
    plt.title('Convergence Generations for Different Np_cap Values')
    plt.grid(True)
    # plt.savefig('robustness_different_Np_caps_generations.pdf')
    plt.show()

    # Boxplot for convergence times for different Np_cap values
    plt.figure(figsize=(12, 6))
    plt.boxplot(convergence_times, labels=[f'Np_cap = {Np_cap}' for Np_cap in Np_cap_values])
    plt.xlabel('Np_cap Values')
    plt.ylabel('Time Taken (s)')
    plt.title('Convergence Time for Different Np_cap Values')
    plt.grid(True)
    # plt.savefig('robustness_different_Np_caps_time_taken.pdf')
    plt.show()

# Run the robustness tests
if __name__ == "__main__":
    print("Running Test 1: Multiple Random Initializations...")
    test_random_initializations()
    print("Running Test 2: Extreme Mutation Rates...")
    test_extreme_mutation_rates()
    print("Running Test 3: Local Optima Test...")
    test_local_optima()
    print("Running Test 4: Np_caps Test...")
    test_multiple_Np_caps_boxplots()
