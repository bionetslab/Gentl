import numpy as np
from scipy.spatial import distance

def euclidean_distance(goal, chromosome):
    # # Calculate Euclidean dist using numpy:
    # dist=np.linalg.norm(goal-chromosome)
    # Calculate Euclidean dist using scipy:
    dist=distance.euclidean(goal, chromosome)
    return dist

def parent_selection_fitness_evaluation(goal, chromosome):
    # Fitness function -  Euclidean distance: # dist=((x1-x2)^2 + (y1-y2)^2)^(1/2)                   
    euclidean_dist=euclidean_distance(np.array(goal), np.array(chromosome))
    
    # Fitness function -  Manhattan distance: # dist=((x1-x2)^2 + (y1-y2)^2)^(1/2)                   
    euclidean_dist=euclidean_distance(np.array(goal), np.array(chromosome))
    
    
    return euclidean_dist,