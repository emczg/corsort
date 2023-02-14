import numpy as np
from corsort.cor_sort import CorSort
from corsort.sort_quick import SortQuick


class CorSortDelegate(CorSort):
    """
    CorSort that delegates the choice of pairwise comparisons to another sorting algorithm.

    Parameters
    ----------
    sort: Sort
        A sorting algorithm.

    Examples
    --------
        >>> corsort = CorSortDelegate(SortQuick())
        >>> corsort(np.array(['e', 'b', 'a', 'c', 'd'])).n_comparisons_
        8
        >>> corsort.history_comparisons_
        [(1, 0), (2, 0), (3, 0), (4, 0), (2, 1), (1, 3), (1, 4), (3, 4)]
        >>> corsort.__name__
        'corsort_delegate_quicksort'
    """

    def __init__(self, sort):
        super().__init__()
        self.sort = sort
        self.__name__ = "corsort_delegate_" + self.sort.__name__

    def next_compare(self):
        self.sort(self.perm_)
        return self.sort.history_comparisons_
