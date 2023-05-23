import numpy as np
from corsort.sort import Sort
from corsort.distance_to_sorted_array import distance_to_sorted_array


class SortBinaryInsertion(Sort):
    """
    Binary insertion sort.

    Examples
    --------
        >>> binary_insertion_sort = SortBinaryInsertion(compute_history=True)
        >>> my_xs = np.array([4, 1, 7, 6, 0, 8, 2, 3, 5])
        >>> binary_insertion_sort(my_xs).n_comparisons_
        19
        >>> binary_insertion_sort.history_comparisons_  # doctest: +NORMALIZE_WHITESPACE
        [(1, 0), (1, 2), (0, 2), (0, 3), (3, 2), (4, 0), (4, 1), (0, 5), (3, 5), (2, 5),
        (6, 0), (4, 6), (1, 6), (7, 0), (1, 7), (6, 7), (7, 8), (8, 3), (0, 8)]
        >>> binary_insertion_sort.history_comparisons_values_  # doctest: +NORMALIZE_WHITESPACE
        [(1, 4), (1, 7), (4, 7), (4, 6), (6, 7), (0, 4), (0, 1), (4, 8), (6, 8), (7, 8),
        (2, 4), (0, 2), (1, 2), (3, 4), (1, 3), (2, 3), (3, 5), (5, 6), (4, 5)]
        >>> binary_insertion_sort.history_distances_
        [17, 16, 16, 16, 16, 15, 15, 11, 11, 11, 11, 11, 11, 7, 7, 7, 3, 3, 3, 0]
        >>> binary_insertion_sort.sorted_list_
        array([0, 1, 2, 3, 4, 5, 6, 7, 8])

        >>> np.random.seed(42)
        >>> binary_insertion_sort = SortBinaryInsertion(compute_history=False)
        >>> binary_insertion_sort(np.random.permutation(100)).n_comparisons_
        537
    """

    __name__ = 'binary_insertion_sort'

    def __init__(self, compute_history=False):
        """
        Examples
        --------
        Before using the algorithm, `history_comparisons_values_` is None:

            >>> binary_insertion_sort = SortBinaryInsertion()
            >>> print(binary_insertion_sort.history_comparisons_values_)
            None
        """
        super().__init__(compute_history=compute_history)
        self.sorted_indices_ = None

    def _initialize_algo_aux(self):
        self.sorted_indices_ = np.arange(self.n_)

    def _call_aux(self):
        _binary_insertion_sort(self.sorted_indices_, lt=self.test_i_lt_j)

    def distance_to_sorted_array(self):
        return distance_to_sorted_array(self.perm_[self.sorted_indices_])

    @property
    def sorted_list_(self):
        return self.perm_[self.sorted_indices_]


def _binary_insertion_sort(xs, lt=None):
    """
    Binary insertion sort.

    Sort the array in place.

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
        >>> _binary_insertion_sort(my_xs)
        >>> my_xs
        array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    """
    if lt is None:
        def lt(x, y):
            return x < y
    n = len(xs)
    for i_to_sort in range(n):
        x_to_sort = xs[i_to_sort]
        i_smaller_or_equal = -1
        i_greater = i_to_sort
        while i_greater - i_smaller_or_equal > 1:
            i_test = (i_smaller_or_equal + i_greater) // 2
            if lt(x_to_sort, xs[i_test]):
                i_greater = i_test
            else:
                i_smaller_or_equal = i_test
        xs[i_greater + 1: i_to_sort + 1] = xs[i_greater: i_to_sort]
        xs[i_greater] = x_to_sort
