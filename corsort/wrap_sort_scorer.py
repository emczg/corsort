import numpy as np
from corsort.distance_to_sorted_array import distance_to_sorted_array
from corsort.sort_quick import SortQuick
from corsort.jit_scorers import jit_scorer_rho


class WrapSortScorer:
    """
    Examples
    --------
        >>> my_sort = SortQuick(compute_history=False)
        >>> jit_sort = WrapSortScorer(scorer=jit_scorer_rho, sort=my_sort, compute_history=False)
        >>> my_xs = np.array([4, 1, 7, 6, 0, 8, 2, 3, 5])
        >>> jit_sort(my_xs).n_comparisons_
        16
        >>> jit_sort.__name__
        'quicksort_rho'
        >>> jit_sort.history_comparisons_  # doctest: +NORMALIZE_WHITESPACE
        [(1, 0), (0, 2), (0, 3), (4, 0), (0, 5), (6, 0), (7, 0), (0, 8),
        (4, 1), (1, 6), (1, 7), (6, 7), (3, 2), (2, 5), (8, 2), (8, 3)]
        >>> jit_sort.history_comparisons_values_  # doctest: +NORMALIZE_WHITESPACE
        [(1, 4), (4, 7), (4, 6), (0, 4), (4, 8), (2, 4), (3, 4), (4, 5),
        (0, 1), (1, 2), (1, 3), (2, 3), (6, 7), (7, 8), (5, 7), (5, 6)]
    """

    def __init__(self, scorer, sort, compute_history=False):
        """
        Examples
        --------
        Before using the algorithm, `history_comparisons_values_` is None:

            >>> my_sort = SortQuick()
            >>> jit_sort = WrapSortScorer(scorer=jit_scorer_rho, sort=my_sort)
            >>> print(jit_sort.history_comparisons_values_)
            None
        """
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

    @property
    def history_comparisons_values_(self):
        """:class:`list` of :class:`tuple`: History of the pairwise comparisons, in terms of compared values.
        Tuple (x, y) means that items of values x and y were compared, and that x < y.
        """
        if self.history_comparisons_ is None:
            return None
        return [(self.perm_[i], self.perm_[j]) for (i, j) in self.history_comparisons_]
