import numpy as np
from corsort.sort import Sort
from corsort.distance_to_sorted_array import distance_to_sorted_array


class SortShell(Sort):
    """
    Shellsort.

    Examples
    --------
        >>> shellsort = SortShell(compute_history=True)
        >>> my_xs = np.array([4, 1, 7, 6, 0, 8, 2, 3, 5])
        >>> shellsort(my_xs).n_comparisons_
        22
        >>> shellsort.history_comparisons_  # doctest: +NORMALIZE_WHITESPACE
        [(4, 0), (4, 5), (1, 5), (6, 2), (6, 7), (7, 3), (7, 8), (0, 8), (4, 1), (1, 6),
        (6, 7), (7, 0), (0, 5), (2, 5), (0, 2), (3, 5), (3, 2), (0, 3), (8, 5), (8, 2), (8, 3), (0, 8)]
        >>> shellsort.history_comparisons_values_  # doctest: +NORMALIZE_WHITESPACE
        [(0, 4), (0, 8), (1, 8), (2, 7), (2, 3), (3, 6), (3, 5), (4, 5), (0, 1), (1, 2),
        (2, 3), (3, 4), (4, 8), (7, 8), (4, 7), (6, 8), (6, 7), (4, 6), (5, 8), (5, 7), (5, 6), (4, 5)]
        >>> shellsort.history_distances_
        [17, 22, 14, 13, 9, 8, 7, 6, 6, 6, 6, 6, 3, 5, 3, 3, 3, 2, 0, 0, 0, 0, 0]
        >>> shellsort.sorted_list_
        array([0, 1, 2, 3, 4, 5, 6, 7, 8])

        >>> np.random.seed(42)
        >>> shellsort = SortShell(compute_history=False)
        >>> shellsort(np.random.permutation(100)).n_comparisons_
        774
    """

    __name__ = 'shellsort'

    def __init__(self, compute_history=False, gap_sequence=None):
        """
        Examples
        --------
        Before using the algorithm, `history_comparisons_values_` is None:

            >>> shellsort = SortShell()
            >>> print(shellsort.history_comparisons_values_)
            None
        """
        super().__init__(compute_history=compute_history)
        if gap_sequence is None:
            gap_sequence = [701, 301, 132, 57, 23, 10, 4, 1]  # Ciura gap sequence
        self.gap_sequence = gap_sequence
        self.sorted_indices_ = None

    def _initialize_algo_aux(self):
        self.sorted_indices_ = np.arange(self.n_)

    def _call_aux(self):
        _shellsort(self.sorted_indices_, gap_sequence=self.gap_sequence, lt=self.test_i_lt_j)

    def distance_to_sorted_array(self):
        return distance_to_sorted_array(self.perm_[self.sorted_indices_])

    @property
    def sorted_list_(self):
        return self.perm_[self.sorted_indices_]


def _shellsort(xs, gap_sequence, lt=None):
    """
    Inspired by https://en.wikipedia.org/wiki/Shellsort

    Sort the array in place.

    Parameters
    ----------
    xs: :class:`~numpy.ndarray`
        Array to sort.
    gap_sequence: list
        Gap sequence.
    lt: callable
        lt(x, y) is the test used to determine whether element x is lower than y.
        Default: operator "<".

    Examples
    --------
        >>> gap_sequence = [701, 301, 132, 57, 23, 10, 4, 1]
        >>> my_xs = np.array([4, 1, 7, 6, 0, 8, 2, 3, 5])
        >>> _shellsort(my_xs, gap_sequence)
        >>> my_xs
        array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    """
    # Set optional arguments.
    if lt is None:
        def lt(x, y):
            return x < y

    n = len(xs)
    for gap in gap_sequence:
        for i in range(gap, n):
            temp = xs[i]
            for j in range(i, -1, -gap):
                xs[j] = xs[j - gap]
                if lt(xs[j - gap], temp):
                    break
            xs[j] = temp
