import random

def mutate(offspring, mutation_rate, g):
    """
    Perform mutation on the offspring.

    Parameters:
    offspring: The list of offspring chromosomes.
    mutation_rate: The probability of mutation for each gene.
    g: The gene pool.

    Returns:
    mutated_offspring: The list of mutated offspring chromosomes.
    """
    mutated_offspring = []
    for chromosome in offspring:
        for i in range(len(chromosome)):
            if random.random() < mutation_rate:
                chromosome[i] = random.choice(g)
        mutated_offspring.append(chromosome)
    return mutated_offspring
