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
        [55, 42, 51, 49, 49, 49, 43, 37, 36, 35, 34, 32, 32, 33, 26, 22, 17, 19, 20, 18, 13,
        11, 11, 11, 11, 6, 6, 7, 9, 7, 5, 5, 4, 2, 2, 2, 1, 2, 1]
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
