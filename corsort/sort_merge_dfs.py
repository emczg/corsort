import numpy as np
from corsort.sort import Sort
from corsort.distance_to_sorted_array import distance_to_sorted_array
from corsort.merge import merge


class SortMergeDfs(Sort):
    """
    Merge sort, DFS version (i.e. "natural" recursive version).

    Examples
    --------
        >>> merge_sort = SortMergeDfs(compute_history=True)
        >>> my_xs = np.array([4, 1, 7, 6, 0, 8, 2, 3, 5])
        >>> merge_sort(my_xs).n_comparisons_
        19
        >>> merge_sort.history_comparisons_  # doctest: +NORMALIZE_WHITESPACE
        [(1, 0), (3, 2), (1, 3), (0, 3), (4, 5), (7, 8), (6, 7), (4, 6), (6, 5), (7, 5),
        (8, 5), (4, 1), (1, 6), (6, 0), (7, 0), (0, 8), (8, 3), (3, 5), (2, 5)]
        >>> merge_sort.history_distances_
        [17, 16, 15, 15, 15, 15, 15, 15, 15, 14, 13, 12, 8, 8, 5, 2, 2, 0, 0, 0]
        >>> merge_sort.sorted_list_
        array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    """

    __name__ = 'mergesort_dfs'

    def __init__(self, compute_history=False):
        super().__init__(compute_history=compute_history)
        self.sorted_indices_ = None

    def _initialize_algo_aux(self):
        self.sorted_indices_ = np.arange(self.n_)

    def _call_aux(self):
        _merge_sort_dfs(self.sorted_indices_, lt=self.test_i_lt_j)

    def distance_to_sorted_array(self):
        return distance_to_sorted_array(self.perm_[self.sorted_indices_])

    @property
    def sorted_list_(self):
        return self.perm_[self.sorted_indices_]


def _merge_sort_dfs(collection, lt=None, i=0, j=None):
    """
    Merge sort (DFS).

    Parameters
    ----------
    collection: class: `list`
        A list to sort.
    lt: class: `function`
        A function that takes two elements x and y return the boolean x < y.
    i: :class:`int`
        Index of the left boundary.
    j: :class:`int`
        Index of the right boundary.

    Examples
    --------
        >>> my_xs = [7, 3, 2, 1, 4, 6, 0, 5]
        >>> _merge_sort_dfs(my_xs)
        >>> my_xs
        [0, 1, 2, 3, 4, 5, 6, 7]

        >>> my_xs = [4, 1, 7, 6, 0, 8, 2, 3, 5]
        >>> _merge_sort_dfs(my_xs)
        >>> my_xs
        [0, 1, 2, 3, 4, 5, 6, 7, 8]

        >>> my_xs = np.array([4, 1, 7, 6, 0, 8, 2, 3, 5])
        >>> indices = np.arange(9)
        >>> def my_lt(my_i, my_j):
        ...     return my_xs[my_i] < my_xs[my_j]
        >>> _merge_sort_dfs(indices, my_lt)
        >>> my_xs[indices]
        array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    """
    if lt is None:
        def lt(x, y):
            return x < y
    if j is None:
        j = len(collection)
    if j - i > 1:
        middle = (i + j) // 2
        _merge_sort_dfs(collection, lt, i, middle)
        _merge_sort_dfs(collection, lt, middle, j)
        merge(collection, i, middle, j, lt=lt)
