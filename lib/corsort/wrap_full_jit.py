import numpy as np
from corsort.entropy_bound import entropy_bound
from corsort.distance_to_sorted_array import distance_to_sorted_array
from corsort.jit_sorts import jit_corsort_borda, \
    jit_corsort_delta_max_rho, jit_corsort_delta_sum_rho, jit_corsort_delta_max_delta, jit_corsort_delta_sum_delta, \
    jit_corsort_rho_max_rho, jit_corsort_rho_sum_rho, jit_corsort_rho_max_delta, jit_corsort_rho_sum_delta, \
    jit_heapsort


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
    history_states_: :class:`list` of :class:`tuple`
        History of the state of the list.

    Examples
    --------
        >>> corsort = WrapFullJit(jit_sort=jit_corsort_borda, compute_history=True, record_states=True)
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
        >>> corsort.history_comparisons_  # doctest: +NORMALIZE_WHITESPACE
        [(0, 1), (3, 2), (3, 0), (0, 4), (4, 1), (2, 4), (2, 0), (0, 5), (7, 6), (7, 0),
        (3, 7), (0, 8), (8, 5), (5, 1), (8, 6), (2, 7), (9, 10), (8, 10), (11, 8), (11, 7),
        (13, 12), (8, 12), (12, 10), (6, 12), (4, 6), (12, 1), (10, 1), (13, 0), (2, 13),
        (11, 2), (11, 3), (7, 13), (9, 0), (2, 9), (4, 8), (5, 12), (6, 5), (7, 9), (13, 9),
        (14, 0), (2, 14), (13, 14), (14, 9)]
        >>> corsort.history_comparisons_values_  # doctest: +NORMALIZE_WHITESPACE
        [(7, 14), (1, 2), (1, 7), (7, 8), (8, 14), (2, 8), (2, 7), (7, 11), (3, 10), (3, 7),
        (1, 3), (7, 9), (9, 11), (11, 14), (9, 10), (2, 3), (6, 13), (9, 13), (0, 9), (0, 3),
        (4, 12), (9, 12), (12, 13), (10, 12), (8, 10), (12, 14), (13, 14), (4, 7), (2, 4),
        (0, 2), (0, 1), (3, 4), (6, 7), (2, 6), (8, 9), (11, 12), (10, 11), (3, 6), (4, 6),
        (5, 7), (2, 5), (4, 5), (5, 6)]

        >>> p = np.array([2, 1, 3, 0])
        >>> corsort(p).history_states_
        [[2, 1, 3, 0], [1, 3, 0, 2], [1, 0, 2, 3], [1, 0, 2, 3], [1, 0, 2, 3], [0, 1, 2, 3]]
    """

    def __init__(self, jit_sort, compute_history=False, record_states=False):
        """
        Examples
        --------
        Before using the algorithm, `history_comparisons_values_` is None:

            >>> corsort = WrapFullJit(jit_sort=jit_corsort_borda, compute_history=False)
            >>> print(corsort.history_comparisons_values_)
            None
        """
        # Parameters
        self.jit_sort = jit_sort
        name_jit_sort = jit_sort.__name__
        i = name_jit_sort.find('jit_')
        if i >= 0:
            name_jit_sort = name_jit_sort[i + 4:]
        self.__name__ = name_jit_sort
        self.compute_history = compute_history
        self.record_states = record_states
        # Computed values
        self.n_ = None
        self.perm_ = None
        self.n_comparisons_ = None
        self.history_distances_ = None
        self.history_comparisons_ = None
        self.history_states_ = None

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
        if self.record_states:
            self.history_states_ = [list(state) for state in states]
        return self

    @property
    def history_comparisons_values_(self):
        """:class:`list` of :class:`tuple`: History of the pairwise comparisons, in terms of compared values.
        Tuple (x, y) means that items of values x and y were compared, and that x < y.
        """
        if self.history_comparisons_ is None:
            return None
        return [(self.perm_[i], self.perm_[j]) for (i, j) in self.history_comparisons_]


class JitCorsortBorda(WrapFullJit):
    """
    Corsort "Borda". Cf. :class:`WrapFullJit`.

    Examples
    --------
        >>> sort = JitCorsortBorda()
    """

    def __init__(self, compute_history=False, record_states=False):
        super().__init__(
            jit_sort=jit_corsort_borda,
            compute_history=compute_history,
            record_states=record_states,
        )


class JitCorsortDeltaMaxRho(WrapFullJit):
    """
    Corsort with delta core scorer, max-knowledge tie-break, and rho output scorer. Cf. :class:`WrapFullJit`.

    Examples
    --------
        >>> sort = JitCorsortDeltaMaxRho()
    """

    def __init__(self, compute_history=False, record_states=False):
        super().__init__(
            jit_sort=jit_corsort_delta_max_rho,
            compute_history=compute_history,
            record_states=record_states,
        )


class JitCorsortDeltaSumRho(WrapFullJit):
    """
    Corsort with delta core scorer, sum-knowledge tie-break, and rho output scorer. Cf. :class:`WrapFullJit`.

    Examples
    --------
        >>> sort = JitCorsortDeltaSumRho()
    """

    def __init__(self, compute_history=False, record_states=False):
        super().__init__(
            jit_sort=jit_corsort_delta_sum_rho,
            compute_history=compute_history,
            record_states=record_states,
        )


class JitCorsortDeltaMaxDelta(WrapFullJit):
    """
    Corsort with delta core scorer, max-knowledge tie-break, and delta output scorer. Cf. :class:`WrapFullJit`.

    Examples
    --------
        >>> sort = JitCorsortDeltaMaxDelta()
    """

    def __init__(self, compute_history=False, record_states=False):
        super().__init__(
            jit_sort=jit_corsort_delta_max_delta,
            compute_history=compute_history,
            record_states=record_states,
        )


class JitCorsortDeltaSumDelta(WrapFullJit):
    """
    Corsort with delta core scorer, sum-knowledge tie-break, and delta output scorer. Cf. :class:`WrapFullJit`.

    Examples
    --------
        >>> sort = JitCorsortDeltaSumDelta()
    """

    def __init__(self, compute_history=False, record_states=False):
        super().__init__(
            jit_sort=jit_corsort_delta_sum_delta,
            compute_history=compute_history,
            record_states=record_states,
        )


class JitCorsortRhoMaxRho(WrapFullJit):
    """
    Corsort with rho core scorer, max-knowledge tie-break, and rho output scorer. Cf. :class:`WrapFullJit`.

    Examples
    --------
        >>> sort = JitCorsortRhoMaxRho()
    """

    def __init__(self, compute_history=False, record_states=False):
        super().__init__(
            jit_sort=jit_corsort_rho_max_rho,
            compute_history=compute_history,
            record_states=record_states,
        )


class JitCorsortRhoSumRho(WrapFullJit):
    """
    Corsort with rho core scorer, sum-knowledge tie-break, and rho output scorer. Cf. :class:`WrapFullJit`.

    Examples
    --------
        >>> sort = JitCorsortRhoSumRho()
    """

    def __init__(self, compute_history=False, record_states=False):
        super().__init__(
            jit_sort=jit_corsort_rho_sum_rho,
            compute_history=compute_history,
            record_states=record_states,
        )


class JitCorsortRhoMaxDelta(WrapFullJit):
    """
    Corsort with rho core scorer, max-knowledge tie-break, and delta output scorer. Cf. :class:`WrapFullJit`.

    Examples
    --------
        >>> sort = JitCorsortRhoMaxDelta()
    """

    def __init__(self, compute_history=False, record_states=False):
        super().__init__(
            jit_sort=jit_corsort_rho_max_delta,
            compute_history=compute_history,
            record_states=record_states,
        )


class JitCorsortRhoSumDelta(WrapFullJit):
    """
    Corsort with rho core scorer, sum-knowledge tie-break, and delta output scorer. Cf. :class:`WrapFullJit`.

    Examples
    --------
        >>> sort = JitCorsortRhoSumDelta()
    """

    def __init__(self, compute_history=False, record_states=False):
        super().__init__(
            jit_sort=jit_corsort_rho_sum_delta,
            compute_history=compute_history,
            record_states=record_states,
        )


class JitHeapsort(WrapFullJit):
    """
    Heapsort. Cf. :class:`WrapFullJit`.

    Examples
    --------
        >>> sort = JitHeapsort()
    """

    def __init__(self, compute_history=False, record_states=False):
        super().__init__(
            jit_sort=jit_heapsort,
            compute_history=compute_history,
            record_states=record_states,
        )
