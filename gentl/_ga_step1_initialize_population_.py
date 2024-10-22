import random

def _ga_step1_initialize_population_(goal, Np_cap, g):
  p = []
  chromosome_length = len(goal) # $l_c$ = $chromosome_length$ = # of genes in each chromosome_length                                             

  for i in range(Np_cap):
      chromosome = []
      for gene in range(chromosome_length):
          chromosome.append(random.choice(g))
      p.append(chromosome)
  return p