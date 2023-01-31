from corsort.CorSort import CorSort
from corsort.SortQuick import SortQuick
from corsort.record_comparisons import record_comparisons


class CorSortDelegate(CorSort):
    """
    CorSort that delegates the choice of pairwise comparisons to another sorting algorithm.

    Parameters
    ----------
    sort: callable
        A sorting algorithm.

    Examples
    --------
        >>> cor_sort_quicksort = CorSortDelegate(SortQuick(compute_history=False))
        >>> cor_sort_quicksort(['e', 'b', 'a', 'c', 'd'])
        8
        >>> cor_sort_quicksort.history_comparisons
        [(1, 0), (2, 0), (3, 0), (4, 0), (2, 1), (3, 1), (4, 1), (4, 3)]

    Notes
    -----
        Currently, this does not work if compute_history is True, because the comparisons used
        to compute the distance are also recorded.
    """

    def __init__(self, sort):
        self.sort = sort
        self.history_comparisons = []
        self.sort_and_record = record_comparisons(self.sort, self.history_comparisons)
        super().__init__()

    def next_compare(self):
        self.sort_and_record(self.perm_)  # Updates self.history_comparisons
        return self.history_comparisons
