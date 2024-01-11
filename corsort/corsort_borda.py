import numpy as np
from corsort.entropy_bound import entropy_bound
from corsort.corsort import Corsort


class CorsortBorda(Corsort):
    """
    Corsort based on the Borda score.

    Examples
    --------
        >>> np.random.seed(22)
        >>> n = 15
        >>> perm = np.random.permutation(n)
        >>> corsort = CorsortBorda(compute_history=True)
        >>> corsort(perm).n_comparisons_
        38
        >>> entropy_bound(n)  # doctest: +ELLIPSIS
        40.24212...
        >>> corsort.history_distances_ # doctest: +NORMALIZE_WHITESPACE
        [76, 64, 76, 74, 74, 72, 64, 58, 56, 54, 56, 52, 58, 56, 44, 36, 28, 30, 30, 28, 22, 20, 20, 20, 18, 10,
         12, 12, 16, 12, 10, 10, 8, 4, 4, 4, 2, 4, 2]
        >>> corsort.__name__
        'corsort_borda'
    """

    __name__ = 'corsort_borda'

    def next_compare(self):
        while True:
            gain_matrix = -np.abs(self.position_estimates_[np.newaxis, :] - self.position_estimates_[:, np.newaxis])
            # (i, j): argmax of the gain, with i < j and leq_[i, j] == 0.
            i, j = np.unravel_index(
                np.argmax(
                    np.where(np.triu(self.leq_ == 0), gain_matrix, -1)
                ),
                gain_matrix.shape
            )
            if i == j == 0:  # pragma: no cover
                break
            else:
                yield i, j
