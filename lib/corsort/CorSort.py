import numpy as np

from corsort import distance_to_sorted_array


class CorSort:
    """
    Attributes
    ----------

    n_: :class:`int`:
        Number of items
    perm_: :class:`~numpy.ndarray`
        Input permutation
    an_: :class:`list` of :class:`set`
        Ancestors
    de_: :class:`list` of :class:`set`
        Descendants
    distances_: :class:`list` of :class:`int`
        KT distance to complete sort
    n_c_: :class:`int`
        Number of comparison performed.
    """
    def __init__(self):
        self.n_ = None
        self.perm_ = None
        self.an_ = None
        self.de_ = None
        self.distances_ = None
        self.n_c_ = None
        self.pos_ = None

    def update_pos(self):
        """
        Update estimated position of each element.

        Returns
        -------
        None
        """
        self.pos_ = np.array([(len(self.de_[i]) - 1 + self.n_ - len(self.an_[i])) / 2 for i in range(self.n_)])

    def test_i_lt_j(self, i, j):
        """

        Parameters
        ----------
        i: :class:`int`
            First index
        j: :class:`int`
            Second index

        Returns
        -------
        :class:`bool`
            The comparison between the two indexed elements.

        """
        return self.perm_[i] < self.perm_[j]

    def apply_i_lt_j(self, i, j):
        """
        Assuming i<j, updates the poset accordingly.

        Parameters
        ----------
        i: :class:`int`
            Index of the small element.
        j: :class:`int`
            Index of the big element

        Returns
        -------
        None
        """
        for jj in self.an_[j]:
            self.de_[jj] |= self.de_[i]
        for ii in self.de_[i]:
            self.an_[ii] |= self.an_[j]
        self.update_pos()

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
        """
        raise NotImplementedError

    def __call__(self, perm):
        """
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
        self.an_ = [{i} for i in range(self.n_)]
        self.de_ = [{i} for i in range(self.n_)]
        self.distances_ = [distance_to_sorted_array(self.perm_)]
        self.update_pos()
        for c in self.next_compare():
            self.compare(*c)
            self.distances_.append(distance_to_sorted_array(
                self.perm_[np.argsort(self.pos_)]))
        self.n_c_ = len(self.distances_) - 1
        return self.n_c_
