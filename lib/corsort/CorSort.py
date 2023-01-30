import numpy as np

from corsort import distance_to_sorted_array


class CorSort:
    """
    CorSort.

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
    sets_ancestors_: :class:`list` of :class:`set`
        For each index (in the original list), set of its ancestors including itself, i.e. items for which we
        know they are greater or equal.
    sets_descendants: :class:`list` of :class:`set`
        For each index (in the original list), set of its descendants including itself, i.e. items for which we
        know they are lower or equal.
    history_distances_: :class:`list` of :class:`int`
        History of the kendall-tau distance to the sorted list.
    n_comparisons_: :class:`int`
        Number of comparison performed.
    position_estimates_: :class:`list` of :class:`float`
        For each index (in the original list), its position estimate in the sorted list.
        Note that a position of 0 means the start of the sorted list, i.e. smallest element, whereas
        a position of `n - 1` means the end of the sorted list, i.e. the greatest element. In other
        words, the position estimates are the Borda scores.
    """
    def __init__(self, compute_history=False):
        # Options
        self.compute_history = compute_history
        # Computed attributes
        self.n_ = None
        self.perm_ = None
        self.sets_ancestors_ = None
        self.sets_descendants = None
        self.history_distances_ = None
        self.n_comparisons_ = None
        self.position_estimates_ = None

    def update_position_estimates(self):
        """
        Update position estimate of each item.

        Examples
        --------
            >>> corsort = CorSort()
            >>> corsort.n_ = 4
            >>> corsort.sets_ancestors_ = [{1, 2, 3}, {}, {}, {1}]
            >>> corsort.sets_descendants = [{}, {0, 2, 3}, {}, {2}]
            >>> corsort.update_position_estimates()
            >>> corsort.position_estimates_
            array([0. , 3. , 1.5, 1.5])
        """
        self.position_estimates_ = np.array([
            (len(self.sets_descendants[i]) - 1 + self.n_ - len(self.sets_ancestors_[i])) / 2
            for i in range(self.n_)
        ])

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

        Examples
        --------
            >>> corsort = CorSort()
            >>> corsort.perm_ = ['b', 'a']
            >>> corsort.test_i_lt_j(0, 1)
            False
        """
        return self.perm_[i] < self.perm_[j]

    def apply_i_lt_j(self, i, j):
        """
        Assuming perm[i] < perm[j], updates the poset accordingly.

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

            >>> corsort.sets_ancestors_ = [{1}, {}, {3}, {}]
            >>> corsort.sets_descendants = [{}, {0}, {}, {2}]

        Assume we learn that perm[1] < perm[2]:

            >>> corsort.apply_i_lt_j(1, 2)
            >>> corsort.sets_ancestors_
            [{1, 3}, {}, {3}, {}]
            >>> corsort.sets_descendants
            [{}, {0}, {}, {0, 2}]

        But that is a bug, because at that stage we should know the full order.
        """
        # TODO: solve this bug.
        for ancestor_of_j in self.sets_ancestors_[j]:
            self.sets_descendants[ancestor_of_j] |= self.sets_descendants[i]
        for descendant_of_i in self.sets_descendants[i]:
            self.sets_ancestors_[descendant_of_i] |= self.sets_ancestors_[j]
        self.update_position_estimates()

    def compare(self, i, j):
        """
        Performs a comparison between i and j.

        Parameters
        ----------
        i: :class:`int`
            First index
        j: :class:`int`
            Second index

        Returns
        -------
        None
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

    def __call__(self, perm):
        """
        Sort.

        Parameters
        ----------
        perm: :class:`numpy.ndarray`
            Input permutation to sort. Typically the output of :meth`~numpy.random.permutation`.

        Returns
        -------
        :class:`int`
            Number of comparisons to sort the permutation.
        """
        if isinstance(perm, list):
            perm = np.array(perm)
        self.n_ = len(perm)
        self.perm_ = perm
        self.sets_ancestors_ = [{i} for i in range(self.n_)]
        self.sets_descendants = [{i} for i in range(self.n_)]
        self.n_comparisons_ = 0
        self.history_distances_ = [distance_to_sorted_array(self.perm_)] if self.compute_history else []
        self.update_position_estimates()
        for c in self.next_compare():
            self.compare(*c)
            self.n_comparisons_ += 1
            if self.compute_history:
                self.history_distances_.append(
                    distance_to_sorted_array(self.perm_[np.argsort(self.position_estimates_)]))
        return self.n_comparisons_
