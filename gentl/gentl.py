import random
from _ga_step1_initialize_population_ import _ga_step1_initialize_population_

def gentl(p, Np_cap, alpha, goal, g, copy=True):
    """Saves i) list of lists as 2D np.array (np.array(list_of_lists)), ii) scores
    OR
    returns i) list of lists, ii) scores.

    Parameters
    ----------
    p: Initial population # Np = len(p) : # of individuals in population during initialization (or, # of chromosomes - one chromosome per individual) [Note that $Np$ is different from actual population $p = list_of_chromosomes_during_initialization$] 
    
    Np_cap: Population cap at every step barring initialization, i.e., # of stronger individuals to be retained at the end of each step.
    
    alpha : rate of mutation
    
    goal : goal list (goal sequence) - here, embedding from cancer ROI
    
    g: set of all permissible values that a particular gene can assume.
    
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
    
    p=_ga_step1_initialize_population_(goal, Np_cap, g)
    
    for p_ in p:
        
