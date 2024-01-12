import numpy as np
from corsort.entropy_bound import entropy_bound
from corsort.corsort import Corsort
from corsort.util_chains import greedy_chain_decomposition


class CorsortChainDecompositionMergeV(Corsort):
    """
    Corsort based on chain decomposition, with "V-shape" merging.

    Examples
    --------
        >>> np.random.seed(22)
        >>> n = 15
        >>> perm = np.random.permutation(n)
        >>> corsort = CorsortChainDecompositionMergeV(compute_history=True)
        >>> corsort(perm).n_comparisons_
        49
        >>> entropy_bound(n)  # doctest: +ELLIPSIS
        40.24212...
        >>> corsort.history_distances_ # doctest: +NORMALIZE_WHITESPACE
        [76, 70, 56, 56, 48, 56, 48, 40, 42, 42, 32, 32, 34, 32, 28, 22, 24, 26, 20, 20, 22, 26, 26, 28, 30, 30, 28, 28,
         28, 30, 28, 32, 26, 32, 24, 24, 18, 18, 10, 10, 8, 8, 8, 8, 6, 6, 2, 2, 2, 0]
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
                elif self.leq_[i, j] == 1:  # pragma: no cover
                    # TODO: Apparently it does not happen during the tests, try to understand why.
                    i_in_chain += 1
                else:
                    j_in_chain += 1
