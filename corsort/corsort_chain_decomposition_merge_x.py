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
        [76, 70, 56, 56, 48, 56, 48, 40, 42, 38, 46, 40, 36, 38, 34, 32, 34, 34, 32, 30, 26, 30, 28, 28, 30, 26, 24, 26,
         26, 26, 24, 26, 28, 28, 26, 20, 20, 16, 10, 10, 8, 8, 8, 8, 6, 4, 2, 0]
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
            except IndexError:  # pragma: no cover
                # One lower branch of the X is exhausted.
                # This should not happen, because we already assumed that an upper branch is exhausted.
                # This would mean, in fact, that the two chains can be fully sorted, a contradiction.
                break
            if self.leq_[i, j] == 0:
                return i, j
            elif self.leq_[i, j] == 1:
                j_in_chain -= 1
            else:  # pragma: no cover
                # TODO: Apparently it does not happen during the tests, try to understand why.
                i_in_chain -= 1
