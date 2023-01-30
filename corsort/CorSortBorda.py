import numpy as np
from corsort import entropy_bound

from corsort.CorSort import CorSort


class CorSortBorda(CorSort):
    """
    Examples
    --------

    >>> np.random.seed(22)
    >>> n_ = 15
    >>> p = np.random.permutation(n_)
    >>> c = CorSortBorda()
    >>> c(p)
    44
    >>> entropy_bound(n_) # doctest: +ELLIPSIS
    40.869...
    >>> c.distances_ # doctest: +NORMALIZE_WHITESPACE
    [55, 42, 51, 49, 49, 48, 40, 39, 33, 29, 29, 28, 28, 28, 28, 26, 26, 21, 20, 16, 14, 11, 10, 9, 8, 7,
     8, 7, 6, 7, 7, 7, 7, 5, 5, 4, 4, 4, 3, 2, 3, 2, 1, 0, 0]
    """

    def gain(self, i, j):
        # TODO: remove this method when it won't be in the parent class anymore.
        raise NotImplementedError

    def next_compare(self):
        while True:
            gain_matrix = -np.abs(self.pos_[np.newaxis, :] - self.pos_[:, np.newaxis])
            arg = None
            gain = None
            for i in range(self.n_):
                for j in range(i + 1, self.n_):
                    if i not in self.de_[j] and i not in self.an_[j]:
                        ng = gain_matrix[i, j]
                        if gain is None or ng > gain:
                            arg = (i, j)
                            gain = ng
            if arg is not None:
                yield arg
            else:
                break
