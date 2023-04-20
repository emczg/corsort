import numpy as np
from corsort.entropy_bound import entropy_bound
from corsort.cor_sort import CorSort
from corsort.util_chains import greedy_chain_decomposition


class CorSortChainDecompositionMergeV(CorSort):
    """
    CorSort based on chain decomposition, with "V-shape" merging.

    Examples
    --------
        >>> np.random.seed(22)
        >>> n = 15
        >>> perm = np.random.permutation(n)
        >>> corsort = CorSortChainDecompositionMergeV(compute_history=True)
        >>> corsort(perm).n_comparisons_
        49
        >>> entropy_bound(n)  # doctest: +ELLIPSIS
        40.24212...
        >>> corsort.history_distances_ # doctest: +NORMALIZE_WHITESPACE
        [55, 50, 38, 37, 33, 39, 33, 29, 31, 30, 22, 22, 23, 20, 17, 14, 16, 18, 15, 15, 16, 17, 17, 18, 19, 17,
        16, 15, 15, 15, 15, 16, 13, 16, 13, 12, 8, 9, 5, 5, 4, 4, 4, 4, 3, 3, 0, 1, 1, 0]
        >>> corsort.__name__
        'corsort_v'
    """

    __name__ = 'corsort_v'

    def next_compare(self):
        while True:
            chains = greedy_chain_decomposition(self.leq_)
            if len(chains) <= 1:
                break
            i_in_chain = 0
            j_in_chain = 0
            found = False
            while not found:
                i = chains[-1][i_in_chain]
                j = chains[-2][j_in_chain]
                if self.leq_[i, j] == 0:
                    found = True
                    yield i, j
                elif self.leq_[i, j] == 1:
                    i_in_chain += 1
                else:
                    j_in_chain += 1