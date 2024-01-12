import numpy as np
from corsort.corsort import Corsort
from corsort.sort_quick import SortQuick
from corsort.scorers import scorer_delta


class CorsortDelegate(Corsort):
    """
    Corsort that delegates the choice of pairwise comparisons to another sorting algorithm.

    Parameters
    ----------
    sort: Sort
        A sorting algorithm.

    Examples
    --------
        >>> corsort = CorsortDelegate(SortQuick(), compute_history=True)
        >>> corsort(np.array(['e', 'b', 'a', 'c', 'd'])).n_comparisons_
        8
        >>> corsort.history_comparisons_
        [(1, 0), (2, 0), (3, 0), (4, 0), (2, 1), (1, 3), (1, 4), (3, 4)]
        >>> corsort.history_distances_
        [8, 2, 2, 2, 2, 4, 2, 0, 0]
        >>> corsort.__name__
        'corsort_delegate_quicksort'
    """

    def __init__(self, sort, compute_history=False, record_leq=False, final_scorer=scorer_delta):
        super().__init__(compute_history=compute_history, record_leq=record_leq, final_scorer=final_scorer)
        self.sort = sort
        self.__name__ = "corsort_delegate_" + self.sort.__name__

    def next_compare(self):
        self.sort(self.perm_)
        return self.sort.history_comparisons_
