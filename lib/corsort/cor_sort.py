import numpy as np

from corsort.distance_to_sorted_array import distance_to_sorted_array
from corsort.sort import Sort


class CorSort(Sort):
    """
    CorSort.

    Parameters
    ----------
    compute_history: :class:`bool`
        If True, then compute the history of the distance to the sorted array.

    Attributes
    ----------
    leq_: :class:`~numpy.ndarray`.
        Matrix of size `(n_, n_)`. Coefficient (i, j) is
        +1 if we know perm_[i] <= perm_[j],
        -1 if we know perm_[i] > perm_[j],
        0 if we do not know the comparison between them.
    position_estimates_: :class:`list` of :class:`float`
        For each index (in the original list), its position estimate in the sorted list.
        Note that a position of 0 means the start of the sorted list, i.e. smallest element, whereas
        a position of `n - 1` means the end of the sorted list, i.e. the greatest element. In other
        words, the position estimates are the Borda scores.

    Notes
    -----
    Cf. also the attributes defined in the parent class :class:`~corsort.Sort`.
    """

    def __init__(self, compute_history=False):
        super().__init__(compute_history=compute_history)
        # Computed attributes
        self.leq_ = None
        self.position_estimates_ = None

    def update_position_estimates(self):
        """
        Update position estimate of each item.

        Examples
        --------
            >>> corsort = CorSort()
            >>> corsort.n_ = 4
            >>> corsort.leq_ = np.array([
            ...     [ 1,  1,  1,  1],
            ...     [-1,  1, -1, -1],
            ...     [-1,  1,  1,  0],
            ...     [-1,  1,  0,  1],
            ... ])
            >>> corsort.update_position_estimates()
            >>> corsort.position_estimates_
            array([0. , 3. , 1.5, 1.5])
        """
        self.position_estimates_ = (np.sum(self.leq_, axis=0) + self.n_) / 2 - 1

    def distance_to_sorted_array(self):
        """
        Distance to sorted array.

        Returns
        -------
        int
            Distance between the current estimation and the sorted array.
        """
        return distance_to_sorted_array(self.perm_[np.argsort(self.position_estimates_)])

    def apply_i_lt_j(self, i, j):
        """
        Assuming perm[i] < perm[j], update the poset accordingly.

        Parameters
        ----------
        i: :class:`int`
            Index of the small item.
        j: :class:`int`
            Index of the big item.

        Examples
        --------
            >>> corsort = CorSort()
            >>> corsort.n_ = 4

        Assume that we know perm[0] < perm[1], and perm[2] < perm[3]:

            >>> corsort.leq_ = np.array([
            ...     [ 1,  1,  0,  0],
            ...     [-1,  1,  0,  0],
            ...     [ 0,  0,  1,  1],
            ...     [ 0,  0, -1,  1],
            ... ])

        Assume we learn that perm[1] < perm[2]:

            >>> corsort.apply_i_lt_j(1, 2)
            >>> corsort.leq_
            array([[ 1,  1,  1,  1],
                   [-1,  1,  1,  1],
                   [-1, -1,  1,  1],
                   [-1, -1, -1,  1]])

        Now we know the full order by transitivity.
        """
        mask_i_and_smaller = self.leq_[:, i] > 0
        mask_j_and_greater = self.leq_[j, :] > 0
        self.leq_[np.ix_(mask_i_and_smaller, mask_j_and_greater)] = 1
        self.leq_[np.ix_(mask_j_and_greater, mask_i_and_smaller)] = -1
        self.update_position_estimates()

    def compare_and_update_poset(self, i, j):
        """
        Perform a comparison between perm[i] and perm[j], and update the poset accordingly.

        Parameters
        ----------
        i: :class:`int`
            First index.
        j: :class:`int`
            Second index.
        """
        if self.test_i_lt_j(i, j):
            self.apply_i_lt_j(i, j)
        else:
            self.apply_i_lt_j(j, i)

    def next_compare(self):
        """
        Iterator of pairwise comparisons to make.

        Returns
        -------
        iterable
            An iterator of pairwise comparisons to make.
        """
        raise NotImplementedError

    def _initialize_algo_aux(self):
        """
        Examples
        --------
            >>> my_sort = CorSort(compute_history=True)
            >>> my_sort._initialize_algo(perm=np.array(['b', 'a']))
            >>> my_sort.test_i_lt_j(0, 1)
            False
            >>> my_sort.n_comparisons_
            1
            >>> my_sort.history_distances_
            [1]
        """
        self.leq_ = np.eye(self.n_, dtype=int)
        self.update_position_estimates()

    def _call_aux(self):
        for c in self.next_compare():
            self.compare_and_update_poset(*c)
        # Final update of history_distance
        if self.compute_history:
            self.history_distances_.append(
                distance_to_sorted_array(self.perm_[np.argsort(self.position_estimates_)]))
        return self
