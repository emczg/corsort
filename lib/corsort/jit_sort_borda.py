from numba import njit
import numpy as np
from corsort.entropy_bound import entropy_bound
from corsort.distance_to_sorted_array import distance_to_sorted_array


@njit
def _jit_corsort(perm):
    n = len(perm)
    leq = np.eye(n, dtype=np.int8)
    pos = np.zeros(n, dtype=np.int_)
    info = np.zeros(n, dtype=np.int_)
    distances = [perm[np.argsort(pos)]]
    comparisons = []
    i, j = 0, 1
    while True:
        diff = n
        xp = 0
        for ii in range(n):
            for jj in range(ii + 1, n):
                if leq[ii, jj] == 0:
                    diff_ij = abs(pos[ii] - pos[jj])
                    if diff_ij > diff:
                        continue
                    xp_ij = info[ii] + info[jj]
                    if diff_ij < diff or (diff_ij == diff and xp_ij > xp):
                        diff = diff_ij
                        xp = xp_ij
                        i, j = ii, jj
        if diff == n:
            break
        if perm[i] > perm[j]:
            i, j = j, i
        comparisons.append((i, j))
        for ii in range(n):
            if leq[ii, i] > 0:
                for jj in range(n):
                    if leq[j, jj] > 0 and leq[ii, jj] == 0:
                        leq[ii, jj] = 1
                        leq[jj, ii] = -1
                        info[ii] += 1
                        info[jj] += 1
                        pos[ii] -= 1
                        pos[jj] += 1
        distances.append(perm[np.argsort(pos)])
    return distances, comparisons


class JitSortBorda:
    """
    A fast, direct, implementation of the Borda Corsort.

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

    >>> np.random.seed(22)
    >>> n = 15
    >>> p = np.random.permutation(n)
    >>> my_sort = JitSortBorda(compute_history=True)
    >>> my_sort(p).n_comparisons_
    43
    >>> entropy_bound(n)  # doctest: +ELLIPSIS
    40.869...
    >>> my_sort.history_distances_  # doctest: +NORMALIZE_WHITESPACE
    [55, 42, 51, 49, 49, 48, 40, 39, 33, 29, 29, 29, 28, 28, 28, 26, 26, 21, 20, 16, 14, 11, 10, 9, 8, 10,
    8, 7, 6, 7, 5, 4, 4, 5, 4, 3, 4, 3, 3, 2, 2, 0, 1, 0]
    >>> my_sort.__name__
    'corsort_borda'
    """

    __name__ = 'corsort_borda'

    def __init__(self, compute_history=False):
        # Parameters
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
        states, comparisons = _jit_corsort(perm)
        self.n_ = len(perm)
        self.perm_ = perm
        self.n_comparisons_ = len(states) - 1
        if self.compute_history:
            self.history_distances_ = [distance_to_sorted_array(state) for state in states]
        else:
            self.history_distances_ = []
        self.history_comparisons_ = comparisons
        return self
