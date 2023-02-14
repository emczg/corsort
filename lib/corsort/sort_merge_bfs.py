import numpy as np
from corsort.sort import Sort
from corsort.distance_to_sorted_array import distance_to_sorted_array
from corsort.merge import merge


class SortMergeBfs(Sort):
    """
    Merge sort, BFS version (i.e. compare pairs first, then quadruples, etc.).

    Examples
    --------
        >>> merge_sort = SortMergeBfs(compute_history=True)
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

    __name__ = 'mergesort_bfs'

    def __init__(self, compute_history=False):
        super().__init__(compute_history=compute_history)
        self.sorted_indices_ = None

    def _initialize_algo_aux(self):
        self.sorted_indices_ = np.arange(self.n_)

    def _call_aux(self):
        _merge_sort_bfs(self.sorted_indices_, lt=self.test_i_lt_j)

    def distance_to_sorted_array(self):
        return distance_to_sorted_array(self.perm_[self.sorted_indices_])

    @property
    def sorted_list_(self):
        return self.perm_[self.sorted_indices_]


def _sub_step(list_indices):
    """
    Compute the indices of the boundaries for the next sub-step of BFS merge sort.

    Parameters
    ----------
    list_indices: :class:`~numpy.ndarray`
        List of indices for the current step.

    Returns
    -------
    :class:`~numpy.ndarray`
        List of indices for the sub-step.

    Examples
    --------
        >>> my_indices = np.array([0, 9])
        >>> my_indices = _sub_step(my_indices)
        >>> my_indices
        array([0, 4, 9])
        >>> my_indices = _sub_step(my_indices)
        >>> my_indices
        array([0, 2, 4, 6, 9])
        >>> my_indices = _sub_step(my_indices)
        >>> my_indices
        array([0, 1, 2, 3, 4, 5, 6, 7, 9])
        >>> my_indices = _sub_step(my_indices)
        >>> my_indices
        array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 9])
    """
    list_indices_sub_step = []
    for i, j in zip(list_indices[:-1], list_indices[1:]):
        list_indices_sub_step.append(i)
        list_indices_sub_step.append((i + j) // 2)
    list_indices_sub_step.append(list_indices[-1])
    return np.array(list_indices_sub_step)


def _lists_indices_steps(n):
    """
    Compute the indices of the boundaries for all the steps of BFS merge sort.

    Parameters
    ----------
    n: :class:`integer`
        Size of the list.

    Returns
    -------
    :class:`list` of :class:`~numpy.ndarray`
        For each step, list of indices for the step.

    Examples
    --------
        >>> _lists_indices_steps(n=9)  # doctest: +NORMALIZE_WHITESPACE
        [array([0, 4, 9]),
        array([0, 2, 4, 6, 9]),
        array([0, 1, 2, 3, 4, 5, 6, 7, 9]),
        array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 9])]
    """
    list_indices = np.array([0, n])
    result = []
    while np.max(list_indices[1:] - list_indices[:-1]) > 1:
        list_indices = _sub_step(list_indices)
        result.append(list_indices)
    return result


def _merge_sort_bfs(collection, lt=None):
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
        >>> _merge_sort_bfs(my_xs)
        >>> my_xs
        [0, 1, 2, 3, 4, 5, 6, 7]

        >>> my_xs = [4, 1, 7, 6, 0, 8, 2, 3, 5]
        >>> _merge_sort_bfs(my_xs)
        >>> my_xs
        [0, 1, 2, 3, 4, 5, 6, 7, 8]

        >>> my_xs = np.array([4, 1, 7, 6, 0, 8, 2, 3, 5])
        >>> indices = np.arange(9)
        >>> def my_lt(my_i, my_j):
        ...     return my_xs[my_i] < my_xs[my_j]
        >>> _merge_sort_bfs(indices, my_lt)
        >>> my_xs[indices]
        array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    """
    n = len(collection)
    lists_indices = _lists_indices_steps(n)
    for list_indices in lists_indices[::-1]:
        for i in range(0, len(list_indices) - 1, 2):
            merge(collection, list_indices[i], list_indices[i+1], list_indices[i+2], lt=lt)
