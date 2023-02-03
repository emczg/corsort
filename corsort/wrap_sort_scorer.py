import numpy as np
from corsort.distance_to_sorted_array import distance_to_sorted_array
from corsort.sort_quick import SortQuick
from corsort.jit_scorers import scorer_spaced


class WrapSortScorer:
    """
    Examples
    --------
        >>> my_sort = SortQuick(compute_history=False)
        >>> jit_sort = WrapSortScorer(scorer=scorer_spaced, sort=my_sort, compute_history=False)
        >>> my_xs = np.array([4, 1, 7, 6, 0, 8, 2, 3, 5])
        >>> jit_sort(my_xs).n_comparisons_
        17
        >>> jit_sort.__name__
        'quicksort_spaced'
    """

    def __init__(self, scorer, sort, compute_history=False):
        # Parameters
        self.scorer = scorer
        self.sort = sort
        self.compute_history = compute_history
        name_scorer = scorer.__name__
        i = name_scorer.find('scorer_')
        if i >= 0:
            name_scorer = name_scorer[i + 7:]
        self.__name__ = self.sort.__name__ + '_' + name_scorer
        # Computed values
        self.n_ = None
        self.perm_ = None
        self.n_comparisons_ = None
        self.history_distances_ = None
        self.history_comparisons_ = None

    def __call__(self, perm):
        if isinstance(perm, list):
            perm = np.array(perm)
        self.n_ = len(perm)
        self.perm_ = perm
        self.sort(perm)
        self.history_comparisons_ = self.sort.history_comparisons_
        downs = np.array([c[0] for c in self.sort.history_comparisons_])
        ups = np.array([c[1] for c in self.sort.history_comparisons_])
        states = self.scorer(self.n_, downs, ups)
        self.n_comparisons_ = states.shape[0] - 1
        if self.compute_history:
            self.history_distances_ = [distance_to_sorted_array(self.perm_[np.argsort(state)])
                                       for state in states]
        else:
            self.history_distances_ = []
        return self
