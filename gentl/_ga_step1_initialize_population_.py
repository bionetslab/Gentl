import random

def _ga_step1_initialize_population_randomly_(goal, Np_initial, g):
    """
       Initialize the population by generating Np_initial chromosomes.
       Each gene value derived from gene pool g.

       Parameters:
       goal: The target list(cancer ROI)
       Np_cap: Population capacity
       g: Gene pool, which contains values from two feature lists.

       Returns:
       p: The initialized population.[[],[],[]]
    """
    p = []
    chromosome_length = len(goal)

    for i in range (int(Np_initial)):
        chromosome = []
        for gene in range(chromosome_length):
            chromosome.append(random.choice(g))
        p.append(chromosome)
    return p