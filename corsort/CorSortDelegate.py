from corsort.CorSort import CorSort
from corsort.SortQuick import SortQuick


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
        >>> corsort(['e', 'b', 'a', 'c', 'd']).n_comparisons_
        8
        >>> corsort.history_comparisons_
        [(1, 0), (2, 0), (3, 0), (4, 0), (2, 1), (1, 3), (1, 4), (3, 4)]
    """

    def __init__(self, sort):
        super().__init__()
        self.sort = sort

    def next_compare(self):
        self.sort(self.perm_)
        return self.sort.history_comparisons_
