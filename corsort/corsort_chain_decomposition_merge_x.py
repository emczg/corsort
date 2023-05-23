import numpy as np
from corsort.entropy_bound import entropy_bound
from corsort.corsort import Corsort
from corsort.util_chains import greedy_chain_decomposition


class CorsortChainDecompositionMergeX(Corsort):
    """
    Corsort based on chain decomposition, with "X-shape" merging.

    Examples
    --------
        >>> np.random.seed(22)
        >>> n = 15
        >>> perm = np.random.permutation(n)
        >>> corsort = CorsortChainDecompositionMergeX(compute_history=True)
        >>> corsort(perm).n_comparisons_
        47
        >>> entropy_bound(n)  # doctest: +ELLIPSIS
        40.24212668333375
        >>> corsort.history_distances_ # doctest: +NORMALIZE_WHITESPACE
        [55, 50, 38, 37, 33, 39, 33, 29, 30, 25, 28, 25, 21, 23, 20, 20, 21, 22, 20, 19, 17, 19, 20,
        19, 19, 16, 16, 17, 17, 17, 15, 16, 17, 16, 15, 11, 11, 9, 5, 5, 4, 4, 4, 4, 3, 2, 1, 0]
        >>> corsort.__name__
        'corsort_x'
    """

    __name__ = 'corsort_x'

    def next_compare(self):
        while True:
            chains = greedy_chain_decomposition(self.leq_)
            if len(chains) <= 1:
                break
            yield self._find_i_j(chains[-1], chains[-2])

    def _find_i_j(self, chain_0, chain_1):
        # Ascending phase (zip the upper branches)
        i_in_chain = len(chain_0) // 2
        j_in_chain = len(chain_1) // 2
        while True:
            try:
                i = chain_0[i_in_chain]
                j = chain_1[j_in_chain]
            except IndexError:  # One upper branch of the X is exhausted.
                break
            if self.leq_[i, j] == 0:
                return i, j
            elif self.leq_[i, j] == 1:
                i_in_chain += 1
            else:
                j_in_chain += 1
        # Descending phase (zip the lower branches)
        i_in_chain = len(chain_0) // 2
        j_in_chain = len(chain_1) // 2
        while True:
            try:
                i = chain_0[i_in_chain]
                j = chain_1[j_in_chain]
            except IndexError:  # One lower branch of the X is exhausted.
                # This should not happen, because we already assumed that an upper branch is exhausted.
                # This would mean, in fact, that the two chains can be fully sorted, a contradiction.
                break
            if self.leq_[i, j] == 0:
                return i, j
            elif self.leq_[i, j] == 1:
                j_in_chain -= 1
            else:
                i_in_chain -= 1
