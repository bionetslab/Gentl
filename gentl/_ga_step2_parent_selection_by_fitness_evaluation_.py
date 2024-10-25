import numpy as np
from scipy.spatial import distance

# def minkowski_distance(goal, chromosome): # p-norm distance
#     # write your code here
#     return dist

# def manhattan_distance(goal, chromosome): # L1-norm distance
#     # write your code here
#     return dist

def euclidean_distance(goal, chromosome): # L2-norm distance
    """
        Calculate the Euclidean distance between the goal and a given chromosome.

        Parameters:
        goal: The target list(cancer ROI)
        chromosome: A chromosome from the population.

        Returns:
        dist: The Euclidean distance between the goal and the chromosome.
        """
    # # Calculate Euclidean dist using numpy:
    # dist=np.linalg.norm(goal-chromosome)
    # Calculate Euclidean dist using scipy:
    dist = distance.euclidean(goal, chromosome)
    return dist

def parent_selection_fitness_evaluation(goal, population):
    """
        Evaluate the fitness of each chromosome.
        Select the top 50% as parents based on fitness (Euclidean distance).

        Parameters:
        goal: The target list(cancer ROI)
        population: List of chromosomes in the population.

        Returns:
        selected_parents: List of selected parents containing the chromosomes closest to the target sequence.
    """

    # Fitness function -  Euclidean distance: # dist=((x1-x2)^2 + (y1-y2)^2)^(1/2)
    #euclidean_dist=euclidean_distance(np.array(goal), np.array(chromosome))
    # Fitness function -  Manhattan distance: # dist=((x1-x2)^2 + (y1-y2)^2)^(1/2)                   
    #euclidean_dist=euclidean_distance(np.array(goal), np.array(chromosome))

    goal_array = np.array(goal)
    # eg: distances = [(chromosome1, 0.1), (chromosome2, 0.2), (chromosome3, 0.3)]
    distances = []
    for chromosome in population:
        dist = euclidean_distance(goal_array, np.array(chromosome))
        distances.append((chromosome, dist))
    distances.sort(key=lambda x: x[1])
    selected_count = len(population) // 2
    selected_parents = [chromosome for chromosome, dist in distances[:selected_count]]
    # eg: selected_parents = [chromosome1, chromosome2]

    return selected_parents
