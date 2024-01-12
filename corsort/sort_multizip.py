import numpy as np
from corsort.sort import Sort
from corsort.distance_to_sorted_array import distance_to_sorted_array
from corsort.multi_merge import multi_merge
from corsort.split_pointer_lists import split_pointer_lists


class SortMultizip(Sort):
    """
    Multizip sort.

    Like bottom-up (BFS) mergesort, we compare pairs first, then quadruples, etc. But at each steps, all merges are
    done in "multizip" style, i.e. one comparison for the first merge, then one for the second merge, etc.

    Examples
    --------
        >>> multizip_sort = SortMultizip(compute_history=True)
        >>> my_xs = np.array([4, 1, 7, 6, 0, 8, 2, 3, 5])
        >>> multizip_sort(my_xs).n_comparisons_
        19
        >>> multizip_sort.history_comparisons_  # doctest: +NORMALIZE_WHITESPACE
        [(7, 8), (1, 0), (3, 2), (4, 5), (6, 7), (1, 3), (4, 6), (0, 3), (6, 5),
        (7, 5), (8, 5), (4, 1), (1, 6), (6, 0), (7, 0), (0, 8), (8, 3), (3, 5), (2, 5)]
        >>> multizip_sort.history_distances_
        [30, 30, 30, 30, 30, 30, 30, 30, 30, 28, 26, 24, 16, 16, 10, 4, 4, 0, 0, 0]
        >>> multizip_sort.sorted_list_
        array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    """

    __name__ = 'multizip_sort'

    def __init__(self, compute_history=False):
        super().__init__(compute_history=compute_history)
        self.sorted_indices_ = None

    def _initialize_algo_aux(self):
        self.sorted_indices_ = np.arange(self.n_)

    def _call_aux(self):
        _multizip_sort(self.sorted_indices_, lt=self.test_i_lt_j)

    def distance_to_sorted_array(self):
        return distance_to_sorted_array(self.perm_[self.sorted_indices_])

    @property
    def sorted_list_(self):
        return self.perm_[self.sorted_indices_]


def _multizip_sort(collection, lt=None):
    """

    Parameters
    ----------
    collection
    lt

    Returns
    -------

    Examples
    --------
        >>> my_xs = [7, 3, 2, 1, 4, 6, 0, 5]
        >>> _multizip_sort(my_xs)
        >>> my_xs
        [0, 1, 2, 3, 4, 5, 6, 7]

        >>> my_xs = [4, 1, 7, 6, 0, 8, 2, 3, 5]
        >>> _multizip_sort(my_xs)
        >>> my_xs
        [0, 1, 2, 3, 4, 5, 6, 7, 8]

        >>> my_xs = np.array([4, 1, 7, 6, 0, 8, 2, 3, 5])
        >>> indices = np.arange(9)
        >>> def my_lt(my_i, my_j):
        ...     return my_xs[my_i] < my_xs[my_j]
        >>> _multizip_sort(indices, my_lt)
        >>> my_xs[indices]
        array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    """
    n = len(collection)
    _split_pointer_lists = split_pointer_lists(n)
    for split_pointer_list in _split_pointer_lists[::-1]:
        multi_merge(collection, split_pointer_list, lt)
