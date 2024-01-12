import numpy as np
from corsort.sort import Sort
from corsort.distance_to_sorted_array import distance_to_sorted_array
from corsort.partition import partition


class SortQuick(Sort):
    """
    Quicksort.

    Examples
    --------
        >>> quicksort = SortQuick(compute_history=True)
        >>> my_xs = np.array([4, 1, 7, 6, 0, 8, 2, 3, 5])
        >>> quicksort(my_xs).n_comparisons_
        16
        >>> quicksort.history_comparisons_  # doctest: +NORMALIZE_WHITESPACE
        [(1, 0), (0, 2), (0, 3), (4, 0), (0, 5), (6, 0), (7, 0), (0, 8),
        (4, 1), (1, 6), (1, 7), (6, 7), (3, 2), (2, 5), (8, 2), (8, 3)]
        >>> quicksort.history_comparisons_values_  # doctest: +NORMALIZE_WHITESPACE
        [(1, 4), (4, 7), (4, 6), (0, 4), (4, 8), (2, 4), (3, 4), (4, 5),
        (0, 1), (1, 2), (1, 3), (2, 3), (6, 7), (7, 8), (5, 7), (5, 6)]
        >>> quicksort.history_distances_
        [30, 30, 30, 30, 24, 24, 16, 8, 8, 6, 6, 6, 6, 6, 6, 2, 0]
        >>> quicksort.sorted_list_
        array([0, 1, 2, 3, 4, 5, 6, 7, 8])

        >>> np.random.seed(42)
        >>> quicksort = SortQuick(compute_history=False)
        >>> quicksort(np.random.permutation(100)).n_comparisons_
        662
    """

    __name__ = 'quicksort'

    def __init__(self, compute_history=False):
        """
        Examples
        --------
        Before using the algorithm, `history_comparisons_values_` is None:

            >>> quicksort = SortQuick()
            >>> print(quicksort.history_comparisons_values_)
            None
        """
        super().__init__(compute_history=compute_history)
        self.sorted_indices_ = None

    def _initialize_algo_aux(self):
        self.sorted_indices_ = np.arange(self.n_)

    def _call_aux(self):
        _quicksort(self.sorted_indices_, lt=self.test_i_lt_j)

    def distance_to_sorted_array(self):
        return distance_to_sorted_array(self.perm_[self.sorted_indices_])

    @property
    def sorted_list_(self):
        return self.perm_[self.sorted_indices_]


def _quicksort(xs, i=0, j=None, lt=None):
    """
    Inspired by https://codereview.stackexchange.com/questions/272639/in-place-quicksort-algorithm-in-python

    Sort the array in place.

    Parameters
    ----------
    xs: :class:`~numpy.ndarray`
        Array to sort.
    i: :class:`int`
        Index of the left boundary.
    j: :class:`int`
        Index of the right boundary.
    lt: callable
        lt(x, y) is the test used to determine whether element x is lower than y.
        Default: operator "<".

    Examples
    --------
        >>> my_xs = np.array([4, 1, 7, 6, 0, 8, 2, 3, 5])
        >>> _quicksort(my_xs)
        >>> my_xs
        array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    """
    # Set optional arguments.
    if j is None:
        j = len(xs) - 1
    if lt is None:
        def lt(x, y):
            return x < y

    # Base case: do nothing if indexes have met or crossed.
    if not i < j:
        return

    # Partition the sequence to enforce the quicksort invariant:
    # "small values" < pivot value <= "large values". The function
    # returns the index of the pivot value.
    pivot_index = partition(xs, i, j, lt)

    # Sort left side and right side.
    _quicksort(xs, i=i, j=pivot_index - 1, lt=lt)
    _quicksort(xs, i=pivot_index + 1, j=j, lt=lt)
