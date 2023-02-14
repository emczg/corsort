import numpy as np
from corsort.entropy_bound import entropy_bound
from corsort.cor_sort import CorSort


class CorSortBorda(CorSort):
    """
    CorSort based on the Borda score.

    Examples
    --------
        >>> np.random.seed(22)
        >>> n = 15
        >>> perm = np.random.permutation(n)
        >>> corsort = CorSortBorda(compute_history=True)
        >>> corsort(perm).n_comparisons_
        44
        >>> entropy_bound(n)  # doctest: +ELLIPSIS
        40.24212...
        >>> corsort.history_distances_ # doctest: +NORMALIZE_WHITESPACE
        [55, 42, 51, 49, 49, 48, 40, 39, 33, 29, 29, 28, 28, 28, 28, 26, 26, 21, 20, 16, 14, 11, 10, 9, 8, 7,
         8, 7, 6, 7, 7, 7, 7, 5, 5, 4, 4, 4, 3, 2, 3, 2, 1, 0, 0, 0]
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
