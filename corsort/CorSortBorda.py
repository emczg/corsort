import numpy as np
from corsort import entropy_bound

from corsort.CorSort import CorSort


class CorSortBorda(CorSort):
    """
    CorSort based on the Borda score.

    Examples
    --------
        >>> np.random.seed(22)
        >>> n = 15
        >>> perm = np.random.permutation(n)
        >>> corsort = CorSortBorda(compute_history=True)
        >>> corsort(perm)
        44
        >>> entropy_bound(n)  # doctest: +ELLIPSIS
        40.869...
        >>> corsort.history_distances_ # doctest: +NORMALIZE_WHITESPACE
        [55, 42, 51, 49, 49, 48, 40, 39, 33, 29, 29, 28, 28, 28, 28, 26, 26, 21, 20, 16, 14, 11, 10, 9, 8, 7,
         8, 7, 6, 7, 7, 7, 7, 5, 5, 4, 4, 4, 3, 2, 3, 2, 1, 0, 0]
    """

    def next_compare(self):
        while True:
            # TODO: remove the `for` loops in there
            gain_matrix = -np.abs(self.position_estimates_[np.newaxis, :] - self.position_estimates_[:, np.newaxis])
            arg = None
            gain = None
            for i in range(self.n_):
                for j in range(i + 1, self.n_):
                    if self.leq_[i, j] == 0:
                        ng = gain_matrix[i, j]
                        if gain is None or ng > gain:
                            arg = (i, j)
                            gain = ng
            if arg is not None:
                yield arg
            else:
                break
