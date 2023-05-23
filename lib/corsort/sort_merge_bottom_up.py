import numpy as np
from corsort.sort import Sort
from corsort.distance_to_sorted_array import distance_to_sorted_array
from corsort.merge import merge
from corsort.split_pointer_lists import split_pointer_lists


class SortMergeBottomUp(Sort):
    """
    Merge sort, bottom-up version (BFS, i.e. compare pairs first, then quadruples, etc.).

    Examples
    --------
        >>> merge_sort = SortMergeBottomUp(compute_history=True)
        >>> my_xs = np.array([4, 1, 7, 6, 0, 8, 2, 3, 5])
        >>> merge_sort(my_xs).n_comparisons_
        19
        >>> merge_sort.history_comparisons_  # doctest: +NORMALIZE_WHITESPACE
        [(7, 8), (1, 0), (3, 2), (4, 5), (6, 7), (1, 3), (0, 3), (4, 6), (6, 5),
        (7, 5), (8, 5), (4, 1), (1, 6), (6, 0), (7, 0), (0, 8), (8, 3), (3, 5), (2, 5)]
        >>> merge_sort.history_distances_
        [17, 17, 16, 15, 15, 15, 15, 15, 15, 14, 13, 12, 8, 8, 5, 2, 2, 0, 0, 0]
        >>> merge_sort.sorted_list_
        array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    """

    __name__ = 'mergesort_bottom_up'

    def __init__(self, compute_history=False):
        super().__init__(compute_history=compute_history)
        self.sorted_indices_ = None

    def _initialize_algo_aux(self):
        self.sorted_indices_ = np.arange(self.n_)

    def _call_aux(self):
        _merge_sort_bottom_up(self.sorted_indices_, lt=self.test_i_lt_j)

    def distance_to_sorted_array(self):
        return distance_to_sorted_array(self.perm_[self.sorted_indices_])

    @property
    def sorted_list_(self):
        return self.perm_[self.sorted_indices_]


def _merge_sort_bottom_up(collection, lt=None):
    """
    Merge sort, bottom-up (BFS).

    Parameters
    ----------
    collection
    lt

    Returns
    -------

    Examples
    --------
        >>> my_xs = [7, 3, 2, 1, 4, 6, 0, 5]
        >>> _merge_sort_bottom_up(my_xs)
        >>> my_xs
        [0, 1, 2, 3, 4, 5, 6, 7]

        >>> my_xs = [4, 1, 7, 6, 0, 8, 2, 3, 5]
        >>> _merge_sort_bottom_up(my_xs)
        >>> my_xs
        [0, 1, 2, 3, 4, 5, 6, 7, 8]

        >>> my_xs = np.array([4, 1, 7, 6, 0, 8, 2, 3, 5])
        >>> indices = np.arange(9)
        >>> def my_lt(my_i, my_j):
        ...     return my_xs[my_i] < my_xs[my_j]
        >>> _merge_sort_bottom_up(indices, my_lt)
        >>> my_xs[indices]
        array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    """
    n = len(collection)
    _split_pointer_lists = split_pointer_lists(n)
    for split_pointer_list in _split_pointer_lists[::-1]:
        for i in range(0, len(split_pointer_list) - 1, 2):
            merge(collection, split_pointer_list[i], split_pointer_list[i+1], split_pointer_list[i+2], lt=lt)
