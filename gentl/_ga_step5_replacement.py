import numpy as np
from scipy.spatial import distance

def replace(population, new_generation, goal):
    """
    Replace the worst-performing individuals in the current population with new offspring, retaining the better-performing individuals.

    Parameters:
    population: The current population.
    new_generation: The new generation of chromosomes.
    goal: The target sequence.

    Returns:
    best_individuals: The updated population containing the best individuals.
    """
    combined_population = population + new_generation
    combined_population.sort(key=lambda chromosome: distance.euclidean(goal, chromosome))
    return combined_population[:len(population)]