import numpy as np
from gentl._ga_step1_initialize_population_ import _ga_step1_initialize_population_randomly_
from gentl._ga_step2_parent_selection_by_fitness_evaluation_ import euclidean_distance, parent_selection_fitness_evaluation
from gentl._ga_step3_crossover import crossover
from gentl._ga_step4_mutation import mutate
from gentl._ga_step5_replacement import replace

def gentl(Np_cap, alpha, goal, g, max_generations=1000, copy=True, fitness_threshold=1.0, mu=None, Np_initial=None, p=[]):
    """Saves i) list of lists as 2D np.array (np.array(list_of_lists)), ii) scores
    OR
    returns i) list of lists, ii) scores.

    Parameters
    ----------
    p: Initial population # Np = len(p) : # of individuals in population during initialization (or, # of chromosomes - one chromosome per individual) [Note that $Np$ is different from actual population $p = list_of_chromosomes_during_initialization$] 
    
    Np_cap: Population cap at every step excluding initialization, i.e., # of stronger individuals to be retained at the end of each step.
    alpha : rate of mutation
    goal : goal list (goal sequence) - here, embedding from cancer ROI
    g: set of all permissible values that a particular gene can assume.
    stopping criteria：
    max_generations: The maximum number of generations to run the genetic algorithm to avoid infinite loops.
    fitness_threshold: The maximum allowable distance from the goal for all individuals in the population to stop the algorithm.

    copy : bool (Optional parameter | default False)
        $copy = True$ returns list of lists, and all scores.
        $copy = False$ saves list of lists, and all scores.
    
    Returns
    -------
    i) list of lists, ii) scores
    
    Note
    ----
    Size of a chromosome (that is, number of genes contained in a chromosome) is constant across all individuals.

    """
    if mu==None or mu==1:
        mu='1'
    elif mu==2 or mu=='random':
        mu='2'
    if mu=='2':
        # Step 1: Initialize population randomly
        if Np_initial==None:
            Np_initial=Np_cap
            population = _ga_step1_initialize_population_randomly_(goal, Np_initial, g)
    elif mu=='1':
        # Step 1: Initialize population provided by user
        if p==None or p==[]:
            if Np_initial==None:
                Np_initial=Np_cap
                population = _ga_step1_initialize_population_randomly_(goal, Np_initial, g)
        else:
            population = p
    
    for generation in range(max_generations):
        # Step 2: Evaluate fitness and select parents
        parents = parent_selection_fitness_evaluation(goal, population)

        # Step 3: Crossover to create offspring
        offspring_size = int(Np_cap - len(parents))
        offspring = crossover(parents, offspring_size, population)

        # Step 4: Mutate the offspring
        mutated_offspring = mutate(offspring, alpha, g)

        # Step 5: Replace the old population with the new generation
        population = replace(population, mutated_offspring, goal)

        # Stopping condition: If all individuals are close enough to the goal
        if all(euclidean_distance(goal, chromosome) < fitness_threshold for chromosome in population):
            #print(f"Solution found in generation {generation}")
            break

    else:
        print(f"Maximum generation limit reached ({max_generations}). Stopping without finding an exact solution.")

    best_individual = min(population, key=lambda chromosome: euclidean_distance(goal, chromosome))
    if copy:
        return best_individual, generation
    else:
        np.save('population.npy', np.array(population))


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