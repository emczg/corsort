import numpy as np


class Sort:
    """
    Abstract class for sorting algorithms.

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
    """

    def __init__(self, compute_history=False):
        # Parameters
        self.compute_history = compute_history
        # Computed values
        self.n_ = None
        self.perm_ = None
        self.n_comparisons_ = None
        self.history_distances_ = None
        self.history_comparisons_ = None

    def distance_to_sorted_array(self):
        """
        Distance to sorted array.

        Returns
        -------
        int
            Distance between the current estimation and the sorted array.
        """
        raise NotImplementedError

    def test_i_lt_j(self, i, j):
        """
        Test whether perm[i] < perm[j].

        Parameters
        ----------
        i: :class:`int`
            First index.
        j: :class:`int`
            Second index.

        Returns
        -------
        :class:`bool`
            True if item of index `i` is lower than item of index `j`.

        Notes
        -----
        The history of distance is computed just *before* the comparison. Hence it should
        be computed a last time at the end of the algorithm.
        """
        self.n_comparisons_ += 1
        if self.compute_history:
            self.history_distances_.append(self.distance_to_sorted_array())
        if self.perm_[i] < self.perm_[j]:
            self.history_comparisons_.append((i, j))
            return True
        else:
            self.history_comparisons_.append((j, i))
            return False

    def _initialize_algo(self, perm):
        """
        Initialize the computed attributes before sorting.

        Parameters
        ----------
        perm: :class:`numpy.ndarray`
            Input permutation to sort. Typically the output of :meth`~numpy.random.permutation`.
        """
        if isinstance(perm, list):
            perm = np.array(perm)
        self.n_ = len(perm)
        self.perm_ = perm
        self.n_comparisons_ = 0
        self.history_distances_ = []
        self.history_comparisons_ = []
        self._initialize_algo_aux()

    def _initialize_algo_aux(self):
        """
        Initialize the computed attributes specific to the implemented algorithm.
        """
        raise NotImplementedError

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
        self._initialize_algo(perm)
        self._call_aux()
        self.history_distances_.append(self.distance_to_sorted_array())
        return self

    def _call_aux(self):
        """
        Must update self.n_comparisons_, self.history_distances_.
        """
        raise NotImplementedError
