import numpy as np
from corsort.entropy_bound import entropy_bound
from corsort.distance_to_sorted_array import distance_to_sorted_array
from corsort.jit_sorts import jit_corsort_borda


class WrapFullJit:
    """
    Delegate everything (sort and scores) to a jit function.

    Parameters
    ----------
    compute_history: :class:`bool`
        If True, then compute the history of the distance to the sorted array.

    Attributes
    ----------
    n_: :class:`int`:
        Number of items in the list.
    perm_: :class:`~numpy.ndarray`
        Input permutation.
    n_comparisons_: :class:`int`
        Number of comparison performed.
    history_distances_: :class:`list` of :class:`int`
        History of the kendall-tau distance to the sorted list.
    history_comparisons_: :class:`list` of :class:`tuple`
        History of the pairwise comparisons. Tuple (i, j) means that items of indices i and j were compared, and
        that perm[i] < perm[j].

    Examples
    --------
        >>> corsort = WrapFullJit(jit_sort=jit_corsort_borda, compute_history=True)
        >>> corsort.__name__
        'corsort_borda'
        >>> np.random.seed(22)
        >>> n = 15
        >>> p = np.random.permutation(n)
        >>> corsort(p).n_comparisons_
        43
        >>> entropy_bound(n)  # doctest: +ELLIPSIS
        40.24212...
        >>> corsort.history_distances_  # doctest: +NORMALIZE_WHITESPACE
        [55, 42, 51, 49, 49, 48, 40, 39, 33, 29, 29, 29, 28, 28, 28, 26, 26, 21, 20, 16, 14, 11, 10, 9, 8, 10,
        8, 7, 6, 7, 5, 4, 4, 5, 4, 3, 4, 3, 3, 2, 2, 0, 1, 0]
    """

    def __init__(self, jit_sort, compute_history=False):
        # Parameters
        self.jit_sort = jit_sort
        name_jit_sort = jit_sort.__name__
        i = name_jit_sort.find('jit_')
        if i >= 0:
            name_jit_sort = name_jit_sort[i + 4:]
        self.__name__ = name_jit_sort
        self.compute_history = compute_history
        # Computed values
        self.n_ = None
        self.perm_ = None
        self.n_comparisons_ = None
        self.history_distances_ = None
        self.history_comparisons_ = None

    def __call__(self, perm):
        """
        Sort.

        Parameters
        ----------
        perm: :class:`numpy.ndarray`
            Input permutation to sort. Typically the output of :meth`~numpy.random.permutation`.

        Returns
        -------
        Itself.
        """
        if isinstance(perm, list):
            perm = np.array(perm)
        states, scores, comparisons = self.jit_sort(perm)
        self.n_ = len(perm)
        self.perm_ = perm
        self.n_comparisons_ = len(states) - 1
        if self.compute_history:
            self.history_distances_ = [distance_to_sorted_array(state) for state in states]
        else:
            self.history_distances_ = []
        self.history_comparisons_ = comparisons
        return self
