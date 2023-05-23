import numpy as np
from itertools import permutations


def transitive_reduction(leq):
    mask_keep = (leq == 1)
    comparisons = [(i, j) for i, j in zip(*np.where(leq == 1)) if i != j]
    for (i, j), (k, l) in permutations(comparisons, 2):
        if j == k:
            mask_keep[i, l] = False
    comparisons = [(i, j) for i, j in zip(*np.where(mask_keep)) if i != j]
    return comparisons
