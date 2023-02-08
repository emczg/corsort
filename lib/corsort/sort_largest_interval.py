import numpy as np
from corsort.sort import Sort
from corsort.distance_to_sorted_array import distance_to_sorted_array
from corsort.partition import partition


class SortLargestInterval(Sort):
    """
    Adaptation of quicksort where we always sort the largest remaining interval.

    Examples
    --------
        >>> my_sort = SortLargestInterval(compute_history=True)
        >>> my_xs = np.array([4, 1, 7, 6, 0, 8, 2, 3, 5])
        >>> my_sort(my_xs).n_comparisons_
        17
        >>> my_sort.history_comparisons_  # doctest: +NORMALIZE_WHITESPACE
        [(1, 0), (0, 2), (0, 3), (4, 0), (0, 5), (6, 0), (7, 0), (0, 8),
        (4, 1), (1, 6), (1, 7), (3, 5), (2, 5), (8, 5), (3, 2), (8, 3), (6, 7)]
        >>> my_sort.history_distances_
        [17, 16, 16, 16, 12, 12, 10, 6, 6, 5, 5, 5, 4, 3, 2, 2, 0, 0]
        >>> my_sort.sorted_list_
        array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    """

    __name__ = 'sort_largest_interval'

    def __init__(self, compute_history=False):
        super().__init__(compute_history=compute_history)
        self.sorted_indices_ = None

    def _initialize_algo_aux(self):
        self.sorted_indices_ = np.arange(self.n_)

    def _call_aux(self):
        _sort_largest_interval(self.sorted_indices_, lt=self.test_i_lt_j)

    def distance_to_sorted_array(self):
        return distance_to_sorted_array(self.perm_[self.sorted_indices_])

    @property
    def sorted_list_(self):
        return self.perm_[self.sorted_indices_]


def _sort_largest_interval(xs, lt=None):
    """
    Adaptation of quicksort where we always sort the largest remaining interval.

    Parameters
    ----------
    xs: :class:`~numpy.ndarray`
        Array to sort.
    lt: callable
        lt(x, y) is the test used to determine whether element x is lower than y.
        Default: operator "<".

    Examples
    --------
        >>> my_xs = np.array([4, 1, 7, 6, 0, 8, 2, 3, 5])
        >>> _sort_largest_interval(my_xs)
        >>> my_xs
        array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    """
    if lt is None:
        def lt(x, y):
            return x < y
    n = len(xs)
    is_well_placed = np.zeros(n, dtype=bool)
    while not np.all(is_well_placed):
        items_well_placed = np.array([-1] + list(np.where(is_well_placed)[0]) + [n])
        widths_intervals = items_well_placed[1:] - items_well_placed[:-1]
        i_interval = np.argmax(widths_intervals)
        bound_before = items_well_placed[i_interval]
        bound_after = items_well_placed[i_interval + 1]
        new_pivot = partition(xs, bound_before + 1, bound_after - 1, lt)
        is_well_placed[new_pivot] = True
