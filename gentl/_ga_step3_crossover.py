import random

def crossover(parents, offspring_size, initial_population):
    """
    Perform crossover operation to generate offspring.

    The first parent is chosen from the top 50% chromosomes.
    The second parent is chosen from the initial population.

    Parameters:
    parents: The selected parents from the top 50% of chromosomes.
    offspring_size: The number of offspring to generate.
    initial_population: The initial population list.

    Returns:
    offspring: The list of generated offspring.
    """
    offspring = []
    for _ in range(int(offspring_size)):
        parent1 = random.choice(parents)
        parent2 = random.choice(initial_population)
        crossover_point = random.randint(1, len(parent1) - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
        offspring.append(child)
        
    return offspring