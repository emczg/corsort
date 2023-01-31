from corsort.CorSort import CorSort
from corsort.quicksort import quicksort
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
        >>> cor_sort_quicksort = CorSortDelegate(quicksort)
        >>> cor_sort_quicksort(['e', 'b', 'a', 'c', 'd'])
        8
        >>> cor_sort_quicksort.history_comparisons
        [(1, 0), (2, 0), (3, 0), (4, 0), (2, 1), (3, 1), (4, 1), (4, 3)]
    """

    def __init__(self, sort):
        self.sort = sort
        self.history_comparisons = []
        self.sort_and_record = record_comparisons(self.sort, self.history_comparisons)
        super().__init__()

    def next_compare(self):
        self.sort_and_record(self.perm_)  # Updates self.history_comparisons
        return self.history_comparisons
