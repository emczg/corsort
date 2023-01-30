import numpy as np
from corsort import entropy_bound
from corsort.CorSortGain import CorSortGain


class CorSortGainLexi(CorSortGain):
    """
    Examples
    --------

    >>> np.random.seed(22)
    >>> n_ = 15
    >>> p = np.random.permutation(n_)
    >>> c = CorSortGainLexi()
    >>> c(p)
    39
    >>> entropy_bound(n_) # doctest: +ELLIPSIS
    40.869...
    >>> c.distances_ # doctest: +NORMALIZE_WHITESPACE
    [55, 42, 51, 49, 49, 43, 39, 38, 37, 36, 28, 27, 22, 21, 20, 19, 13, 14, 13, 15,
    12, 13, 13, 11, 10, 6, 5, 5, 4, 2, 2, 1, 2, 2, 2, 1, 2, 1, 1, 0]
    """

    def gain_i_lt_j(self, i, j):
        """

        Parameters
        ----------
        i: :class:`int`
            Index of the small element.
        j: :class:`int`
            Index of the big element

        Returns
        -------
        :class:`tuple`
            Potential gain if we compare i and j and find that i<j
        """
        gain = 0
        if j in self.an_[i]:
            return 0, 0
        for jj in self.an_[j]:
            gain += len(self.de_[i] - self.de_[jj])
        for ii in self.de_[i]:
            gain += len(self.an_[j] - self.an_[ii])
        return gain, -abs(self.pos_[i] - self.pos_[j])

    def gain(self, i, j):
        """

        Parameters
        ----------
        i: :class:`int`
            First index
        j: :class:`int`
            Second index

        Returns
        -------
        :class:`tuple`
            Ensured gain if i and j are compared.

        """
        return min(self.gain_i_lt_j(i, j), self.gain_i_lt_j(j, i))
