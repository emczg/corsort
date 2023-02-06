import numpy as np
from corsort.entropy_bound import entropy_bound
from corsort.cor_sort_gain import CorSortGain


class CorSortGainLexi(CorSortGain):
    """
    CorSort with lexicographic gain (information gain, difference of position estimates).

    Examples
    --------
        >>> np.random.seed(22)
        >>> n_ = 15
        >>> p = np.random.permutation(n_)
        >>> corsort = CorSortGainLexi(compute_history=True)
        >>> corsort(p).n_comparisons_
        39
        >>> entropy_bound(n_) # doctest: +ELLIPSIS
        40.24212...
        >>> corsort.history_distances_ # doctest: +NORMALIZE_WHITESPACE
        [55, 42, 51, 49, 49, 43, 39, 38, 37, 36, 28, 27, 22, 21, 20, 19, 13, 14, 13, 15,
        12, 13, 13, 11, 10, 6, 5, 5, 4, 2, 2, 1, 2, 2, 2, 1, 2, 1, 1, 0, 0]
        >>> corsort.__name__
        'corsort_lexi'
    """

    __name__ = 'corsort_lexi'

    def gain_i_lt_j(self, i, j):
        """
        Gain if we learn that perm[i] < perm[j].

        Parameters
        ----------
        i: :class:`int`
            Index of the small element.
        j: :class:`int`
            Index of the big element.

        Returns
        -------
        :class:`tuple`
            Potential gain if we compare i and j and find that perm[i] < perm[j].
            First element: number of pairs for which we will learn their comparison if it is the case.
            Second element: difference of position estimate between i and j (currently, i.e. before comparing them).
        """
        gain = 0
        if self.leq_[i, j] == 1:
            return 0, 0
        i_or_lower = (self.leq_[:, i] == 1)
        j_or_greater = (self.leq_[j, :] == 1)
        for ancestor_of_j in np.where(j_or_greater)[0]:
            # ancestor_of_j learns that it's larger than all items i_or_lower, except for those it already knew.
            gain += np.sum(i_or_lower & np.logical_not(self.leq_[:, ancestor_of_j] == 1))
        for descendant_of_i in np.where(i_or_lower)[0]:
            # descendant_of_i learns that it's smaller than all items j_or_greater, except for those it already knew.
            gain += np.sum(j_or_greater & np.logical_not(self.leq_[descendant_of_i, :] == 1))
        return gain, -abs(self.position_estimates_[i] - self.position_estimates_[j])

    def gain(self, i, j):
        """
        Ensured gain when comparing perm[i] and perm[j].

        Parameters
        ----------
        i: :class:`int`
            First index.
        j: :class:`int`
            Second index.

        Returns
        -------
        :class:`tuple`
            Ensured gain if i and j are compared. Cf. :meth:`gain_i_lt_j`.
        """
        return min(self.gain_i_lt_j(i, j), self.gain_i_lt_j(j, i))
