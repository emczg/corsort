import numpy as np
from corsort.distance_to_sorted_array import distance_to_sorted_array


def corsort(perm):
    """
    A fast, direct, implementation of the Borda Corsort.

    Parameters
    ----------
    perm: :class:`numpy.ndarray`
        Input (random) permutation

    Returns
    -------
    d: :class:`int`
        Number of permutations required to sort the input
    distances: :class:`list`
        Evolution of distances to target.

    Examples
    --------

    >>> np.random.seed(22)
    >>> n = 15
    >>> p = np.random.permutation(n)
    >>> d, dists =corsort(p)
    >>> d
    44
    >>> entropy_bound(n) # doctest: +ELLIPSIS
    40.869...
    >>> dists # doctest: +NORMALIZE_WHITESPACE
    [55, 42, 51, 49, 49, 48, 40, 39, 33, 29, 29, 28, 28, 28, 28, 26, 26, 21, 20, 16, 14, 11, 10, 9, 8, 7,
     8, 7, 6, 7, 7, 7, 7, 5, 5, 4, 4, 4, 3, 2, 3, 2, 1, 0, 0]

    """
    n = len(perm)
    an = np.eye(n, dtype=bool)
    pos = np.sum(an, axis=0) - np.sum(an, axis=1)
    distances = [distance_to_sorted_array(perm[np.argsort(pos)])]
    while True:
        pos_matrix = 2*n-np.abs(pos[np.newaxis, :] - pos[:, np.newaxis])
        pos_matrix[an] = 0
        pos_matrix[an.T] = 0
        i, j = np.unravel_index(pos_matrix.argmax(), pos_matrix.shape)
        if pos_matrix[i, j] == 0:
            break
        if perm[i] < perm[j]:
            an[np.ix_(an[:, i], an[j, :])] = True
        else:
            an[np.ix_(an[:, j], an[i, :])] = True
        pos = np.sum(an, axis=0) - np.sum(an, axis=1)
        distances.append(distance_to_sorted_array(perm[np.argsort(pos)]))
    return len(distances)-1, distances


def entropy_bound(n):
    """
    Gives an_ approximation of the information theoretical lower bound of the number of comparisons
    required to sort n_ items.

    An extra offset log2(n_) is added.

    Cf https://en.wikipedia.org/wiki/Comparison_sort

    Parameters
    ----------
    n: :class:`int`
        Number of items to sort.

    Returns
    -------
    :class:`float`
        A lower bound.

    Examples
    --------
    >>> print(f"{entropy_bound(10):.1f}")
    22.1
    >>> print(f"{entropy_bound(100):.1f}")
    526.8
    >>> print(f"{entropy_bound(1000):.1f}")
    8533.1
    """
    return n * (np.log2(n) - 1 / np.log(2))+np.log2(n)


class CorSort:
    """
    Attributes
    ----------

    n_: :class:`int`:
        Nunber of items
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
        self.pos_ =  np.array([(len(self.de_[i]) - 1 + self.n_ - len(self.an_[i])) / 2 for i in range(self.n_)])

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

    def gain(self, i, j):
        raise NotImplementedError

    def apply_i_lt_j(self, i, j):
        """
        Assuming i<j, updates the posset accordingly.

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
        while True:
            gain = self.gain(0, 0)
            arg = None
            for i in range(self.n_):
                for j in range(i + 1, self.n_):
                    ng = self.gain(i, j)
                    if ng > gain:
                        arg = (i, j)
                        gain = ng
            if arg is not None:
                yield arg
            else:
                break

    def __call__(self, perm, comparisons=None):
        """
        Parameters
        ----------
        perm: :class:`numpy.ndarray`
            Input permutation to sort. Typically the output of :meth`~numpy.random.permutation`.
        comparisons: iterator, default=:meth:`next_compare`
            Comparison to make (to feed comparison from an external sort algorithm).

        Returns
        -------
        :class:`int`
            Number of comparisons to sort the permutation.
        """
        if isinstance(perm, list):
            perm = np.array(perm)
        if comparisons is None:
            comparisons = self.next_compare()
        self.n_ = len(perm)
        self.perm_ = perm
        self.an_ = [{i} for i in range(self.n_)]
        self.de_ = [{i} for i in range(self.n_)]
        self.distances_ = [distance_to_sorted_array(self.perm_)]
        self.update_pos()
        for c in comparisons:
            self.compare(*c)
            self.distances_.append(distance_to_sorted_array(
                self.perm_[np.argsort(self.pos_)]))
        self.n_c_ = len(self.distances_) - 1
        return self.n_c_


class CorSortBorda(CorSort):
    """
    Examples
    --------

    >>> np.random.seed(22)
    >>> n_ = 15
    >>> p = np.random.permutation(n_)
    >>> c = CorSortBorda()
    >>> c(p)
    44
    >>> entropy_bound(n_) # doctest: +ELLIPSIS
    40.869...
    >>> c.distances_ # doctest: +NORMALIZE_WHITESPACE
    [55, 42, 51, 49, 49, 48, 40, 39, 33, 29, 29, 28, 28, 28, 28, 26, 26, 21, 20, 16, 14, 11, 10, 9, 8, 7,
     8, 7, 6, 7, 7, 7, 7, 5, 5, 4, 4, 4, 3, 2, 3, 2, 1, 0, 0]
    """

    def next_compare(self):
        while True:
            gain_matrix = -np.abs(self.pos_[np.newaxis, :] - self.pos_[:, np.newaxis])
            arg = None
            gain = None
            for i in range(self.n_):
                for j in range(i + 1, self.n_):
                    if i not in self.de_[j] and i not in self.an_[j]:
                        ng = gain_matrix[i, j]
                        if gain is None or ng > gain:
                            arg = (i, j)
                            gain = ng
            if arg is not None:
                yield arg
            else:
                break


class CorSortLexi(CorSort):
    """
    Examples
    --------

    >>> np.random.seed(22)
    >>> n_ = 15
    >>> p = np.random.permutation(n_)
    >>> c = CorSortLexi()
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
