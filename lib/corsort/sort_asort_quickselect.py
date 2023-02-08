import numpy as np
from corsort.sort import Sort
from corsort.distance_to_sorted_array import distance_to_sorted_array
from corsort.partition import partition


class SortAsortQuickselect(Sort):
    """
    Quicksort.

    Examples
    --------
        >>> asort = SortAsortQuickselect(compute_history=True)
        >>> my_xs = np.array([4, 1, 7, 6, 0, 8, 2, 3, 5])
        >>> asort(my_xs).n_comparisons_
        17
        >>> asort.history_comparisons_  # doctest: +NORMALIZE_WHITESPACE
        [(1, 0), (0, 2), (0, 3), (4, 0), (0, 5), (6, 0), (7, 0), (0, 8),
        (4, 1), (1, 6), (1, 7), (3, 5), (2, 5), (8, 5), (3, 2), (8, 3), (6, 7)]
        >>> asort.history_distances_
        [17, 16, 16, 16, 12, 12, 10, 6, 6, 5, 5, 5, 4, 3, 2, 2, 0, 0]
        >>> asort.sorted_list_
        array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    """

    __name__ = 'asort_quickselect'

    def __init__(self, compute_history=False):
        super().__init__(compute_history=compute_history)
        self.sorted_indices_ = None

    def _initialize_algo_aux(self):
        self.sorted_indices_ = np.arange(self.n_)

    def _call_aux(self):
        _asort_quickselect(self.sorted_indices_, lt=self.test_i_lt_j)

    def distance_to_sorted_array(self):
        return distance_to_sorted_array(self.perm_[self.sorted_indices_])

    @property
    def sorted_list_(self):
        return self.perm_[self.sorted_indices_]


def _pivot_positions(n):
    """
    List the pivot positions: median, then medians of the two halfs, median of the four quarters, etc.

    Parameters
    ----------
    n: :class:`int`
        Size of the list.

    Returns
    -------
    :class:`list`
        The list of the pivots in the desired order.

    Examples
    --------
        >>> _pivot_positions(15)
        [7, 3, 11, 1, 5, 9, 13, 0, 2, 4, 6, 8, 10, 12, 14]

        >>> _pivot_positions(9)
        [4, 1, 6, 0, 2, 5, 7, 3, 8]
    """
    pivots_in_algo_order = []  # What will be returned
    bounds_in_natural_order = [-1, n]  # All pivots plus -1 and n
    set_pivots = {-1, n}
    n_pivots = 0
    while n_pivots < n:
        bounds_in_natural_order_new = []
        for i, j in zip(bounds_in_natural_order[:-1], bounds_in_natural_order[1:]):
            new_pivot = (i + j) // 2
            bounds_in_natural_order_new.append(i)
            bounds_in_natural_order_new.append(new_pivot)
            if new_pivot not in set_pivots:
                pivots_in_algo_order.append(new_pivot)
                set_pivots.add(new_pivot)
                n_pivots += 1
        bounds_in_natural_order_new.append(n)
        bounds_in_natural_order = bounds_in_natural_order_new
    return pivots_in_algo_order


def _asort_quickselect(xs, lt=None):
    """
    ASort algorithm (in place), using quick select for the median.

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
        >>> _asort_quickselect(my_xs)
        >>> my_xs
        array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    """
    if lt is None:
        def lt(x, y):
            return x < y
    n = len(xs)
    positions_pivots_to_find = _pivot_positions(n)
    items_well_placed = {-1, n}
    for position_pivots_to_find in positions_pivots_to_find:
        while position_pivots_to_find not in items_well_placed:
            bound_before = max([
                i for i in items_well_placed
                if i < position_pivots_to_find
            ])
            bound_after = min([
                i for i in items_well_placed
                if i > position_pivots_to_find
            ])
            new_pivot = partition(xs, bound_before + 1, bound_after - 1, lt)
            items_well_placed.add(new_pivot)
