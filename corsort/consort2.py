import numpy as np


def bound(n):
    return n * (np.log2(n) - 1 / np.log(2)) + np.log2(n)


class Node:
    """

    Parameters
    ----------
    v
    n
    """

    def __init__(self, v, n):
        self.v = v
        self.mo = set()  # set of mothers (greater from direct comparison)
        self.da = set()  # set of daughters (lesser from direct comparison)
        self.an = set()  # set of ancestors (ancestors are greater than)
        self.de = set()  # set of descendants (descendants are less than)
        self.u = None
        self.pos = None
        self.refresh(n)

    def __lt__(self, other):
        return self.v < other.v

    def __repr__(self):
        return "Node(" + str(self.v) + ")"

    def refresh(self, n):
        self.u = n - 1 - len(self.an) - len(self.de)
        self.pos = (n - 1 - len(self.an) + len(self.de)) / 2


class CorSort:
    """

    Parameters
    ----------
    perm: :class:`~numpy.ndarray`
        Output of np.random.permutation

    Examples
    --------

    >>> np.random.seed(22)
    >>> n = 15
    >>> p = np.random.permutation(n)
    >>> c = CorSort(p)
    >>> c.sort()
    40
    >>> bound(n) # doctest: +ELLIPSIS
    40.869...
    """
    def __init__(self, perm):
        self.n = len(perm)
        self.xs = [Node(v, self.n) for v in perm]

    def __repr__(self):
        return str([node.v for node in self.xs])

    def gain_i_lt_j(self, i, j):
        """
        Under the hypothesis that ni < nj, evaluate the uncertainty loss.

        Parameters
        ----------
        i: :class:`int`
            Index of the smallest node
        j: :class:`int`
            Index of the biggest node

        Returns
        -------
        :class:`int`
            Potential gain.
        """
        ni = self.xs[i]
        nj = self.xs[j]
        if i in nj.de or i == j:
            return 0
        gain = len(nj.an - ni.an) + 1
        gain += len(ni.de - nj.de) + 1
        for k in nj.an:
            gain += len(ni.de - self.xs[k].de) + 1
        for k in ni.de:
            gain += len(nj.an - self.xs[k].an) + 1
        return gain

    def gain(self, i, j):
        """
        Minimal gain from comparing i and j.

        Parameters
        ----------
        i: :class:`int`
            Node index
        j: :class:`int`
            Node index

        Returns
        -------
        :class:`int`
            Guaranteed gain.
        """
        return min(self.gain_i_lt_j(i, j), self.gain_i_lt_j(j, i))

    def comp(self, i, j):
        """
        Performs comparison (and update accordingly)

        Parameters
        ----------
        i: :class:`int`
            Node index
        j: :class:`int`
            Node index

        Returns
        -------
        None
        """
        ni = self.xs[i]
        nj = self.xs[j]
        if ni < nj:
            ni.mo.add(j)
            ni.an.add(j)
            ni.an = ni.an | nj.an
            nj.da.add(i)
            nj.de.add(i)
            nj.de = nj.de | ni.de
            for k in nj.an:
                self.xs[k].de.add(i)
                self.xs[k].de = self.xs[k].de | ni.de
            for k in ni.de:
                self.xs[k].an.add(j)
                self.xs[k].an = self.xs[k].an | nj.an
            self.refresh()
        else:
            self.comp(j, i)

    def next_comp(self):
        """

        Returns
        -------

        :class:`tuple` or None
            Next pair to compare.
        """
        gain = 0
        arg = None
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if i not in self.xs[j].de and i not in self.xs[j].an:
                    ng = self.gain(i, j)
                    if ng > gain:
                        arg = (i, j)
                        gain = ng
        return arg

    def refresh(self):
        for node in self.xs:
            node.refresh(self.n)

    def sort(self):
        nc = 0
        while c := self.next_comp():
            self.comp(c[0], c[1])
            nc += 1
        return nc

