import string
from fractions import Fraction
from itertools import combinations
from math import comb

import networkx as nx  # type: ignore
import numpy as np
from scipy.optimize import linear_sum_assignment
import svvamp  # type: ignore


class ChainAndY:
    """
    A poset consisting of a chain and a Y-shape.

    Use :meth:`draw` to visualize an example.

    Parameters
    ----------
    a: int
        Number of nodes in the isolated chain.
    b: int
        Number of nodes in the trunk of the Y.
    c, d: int
        Number of nodes in each branch of the Y.
    """

    def __init__(self, a, b, c, d):
        self.a, self.b, self.c, self.d = a, b, c, d
        # Cached variables
        self._cache_average_height_c = None
        self._cache_average_height_d = None
        self._cache_profile_linear_extensions = None
        self._cache_profile_linear_extensions_svvamp = None
        self._cache_positions_counts_in_extensions = None

    @property
    def n_nodes(self):
        """
        Number of nodes

        Returns
        -------
        int
            Number of nodes

        Examples
        --------
            >>> ChainAndY(2, 1, 1, 2).n_nodes
            6
        """
        return self.a + self.b + self.c + self.d

    @property
    def graph(self):
        """
        The poset as a networkx graph.

        The main purpose of the property is to be used by :meth:`draw`.

        Returns
        -------
        class:`~networkx.DiGraph`
            The graph.

        Examples
        --------
            >>> print(ChainAndY(10, 4, 5, 7).graph)
            DiGraph with 26 nodes and 24 edges
        """
        g = nx.DiGraph()
        for i in range(self.a):
            g.add_node(i, pos=(0, i))
            if i != 0:
                g.add_edge(i, i - 1)
        for i in range(self.b):
            g.add_node(self.a + i, pos=(1.5, i))
            if i != 0:
                g.add_edge(self.a + i, self.a + i - 1)
        for i in range(self.c):
            g.add_node(self.a + self.b + i, pos=(1, self.b + i))
            if i == 0:
                g.add_edge(self.a + self.b, self.a + self.b - 1)
            else:
                g.add_edge(self.a + self.b + i, self.a + self.b + i - 1)
        for i in range(self.d):
            g.add_node(self.a + self.b + self.c + i, pos=(2, self.b + i))
            if i == 0:
                g.add_edge(self.a + self.b + self.c, self.a + self.b - 1)
            else:
                g.add_edge(self.a + self.b + self.c + i, self.a + self.b + self.c + i - 1)
        return g

    def draw(self, with_labels=False, alpha_labels=False):
        """
        Draw the poset.

        Parameters
        ----------
        with_labels: bool
            If True, nodes are labeled.
        alpha_labels: bool
            If True, nodes are labeled with letters (there must be 26 nodes at most). In that case, the input
            parameter `with_labels` is ignored and automatically set to True.

        Examples
        --------
            >>> ChainAndY(10, 4, 5, 7).draw()

            >>> ChainAndY(10, 4, 5, 7).draw(alpha_labels=True)
        """
        graph = self.graph
        pos = nx.get_node_attributes(graph, 'pos')
        kwargs = {}
        if alpha_labels:
            with_labels = True
            kwargs['labels'] = dict(zip(graph, string.ascii_lowercase))
        if with_labels:
            kwargs['node_color'] = 'white'
            kwargs['edgecolors'] = 'black'
        nx.draw(self.graph, pos, with_labels=with_labels, **kwargs)

    @property
    def nb_linear_extensions(self):
        """
        Number of linear extensions.

        Returns
        -------
        int
            Number of linear extensions of the poset.

        Examples
        --------
            >>> ChainAndY(10, 4, 5, 7).nb_linear_extensions
            4206894120
        """
        return comb(self.c + self.d, self.c) * comb(self.n_nodes, self.a)

    @property
    def nb_ancestors(self):
        """
        Number of ancestors for each node.

        Returns
        -------
        :class:`~numpy.ndarray`
            Size `a + b + c + d`. Number of ancestors for each node.

        Examples
        --------
            >>> ChainAndY(10, 4, 5, 7).nb_ancestors  # doctest: +NORMALIZE_WHITESPACE
            array([10,  9,  8,  7,  6,  5,  4,  3,  2,  1, 16, 15, 14, 13,  5,  4,  3,
            2,  1,  7,  6,  5,  4,  3,  2,  1])
        """
        return np.concatenate((
            np.array(range(self.a, 0, -1), dtype=int),
            np.array(range(self.b, 0, -1), dtype=int) + self.c + self.d,
            np.array(range(self.c, 0, -1), dtype=int),
            np.array(range(self.d, 0, -1), dtype=int),
        ))

    @property
    def nb_descendants(self):
        """
        Number of ancestors for each node.

        Returns
        -------
        :class:`~numpy.ndarray`
            Size `a + b + c + d`. Number of descendants for each node.

        Examples
        --------
            >>> ChainAndY(10, 4, 5, 7).nb_descendants  # doctest: +NORMALIZE_WHITESPACE
            array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10,  1,  2,  3,  4,  5,  6,  7,
                    8,  9,  5,  6,  7,  8,  9, 10, 11])
        """
        return np.concatenate((
            np.array(range(1, self.a + 1), dtype=int),
            np.array(range(1, self.b + 1), dtype=int),
            np.array(range(1, self.c + 1), dtype=int) + self.b,
            np.array(range(1, self.d + 1), dtype=int) + self.b,
        ))

    @property
    def delta(self):
        """
        Estimator of position "delta" for each node.

        It is the number of descendants, minus the number of ancestors. Up to an affine transformation,
        it is the average between the worst and the best possible positions of the node in the sorted list.

        Returns
        -------
        :class:`~numpy.ndarray`
            Size `a + b + c + d`. Estimator delta for each node.

        Examples
        --------
            >>> ChainAndY(10, 4, 5, 7).delta  # doctest: +NORMALIZE_WHITESPACE
            array([ -9,  -7,  -5,  -3,  -1,   1,   3,   5,   7,   9, -15, -13, -11,
                    -9,   0,   2,   4,   6,   8,  -2,   0,   2,   4,   6,   8,  10])
        """
        return self.nb_descendants - self.nb_ancestors

    @property
    def rho(self):
        """
        Estimator of position "rho" for each node.

        It is `de / (de + an)`, where `de` denotes the number of descendants and `an` the number of ancestors.
        Up to an affine transformation, it would be the average position of the node if its descendants and ancestors
        were forming a chain and if no information was known about the other elements.

        Returns
        -------
        :class:`~numpy.ndarray`
            Size `a + b + c + d`. Estimator rho for each node.

        Examples
        --------
            >>> ChainAndY(10, 4, 5, 7).rho  # doctest: +NORMALIZE_WHITESPACE
            array([Fraction(1, 11), Fraction(2, 11), Fraction(3, 11), Fraction(4, 11),
                   Fraction(5, 11), Fraction(6, 11), Fraction(7, 11), Fraction(8, 11),
                   Fraction(9, 11), Fraction(10, 11), Fraction(1, 17),
                   Fraction(2, 17), Fraction(3, 17), Fraction(4, 17), Fraction(1, 2),
                   Fraction(3, 5), Fraction(7, 10), Fraction(4, 5), Fraction(9, 10),
                   Fraction(5, 12), Fraction(1, 2), Fraction(7, 12), Fraction(2, 3),
                   Fraction(3, 4), Fraction(5, 6), Fraction(11, 12)], dtype=object)
        """
        nb_desc = self.nb_descendants
        nb_anc = self.nb_ancestors
        return np.array([
            Fraction(int(descendants), int(descendants + ancestors))
            for descendants, ancestors in zip(nb_desc, nb_anc)
        ])

    @property
    def _average_normalized_height_a(self):
        """
        Average normalized height for chain `a`.

        Returns
        -------
        :class:`~numpy.ndarray`
            Size `a`. Average normalized height for each node.

        Examples
        --------
        Since it depends only on `a`, it is not necessary to give the other parameters:

            >>> ChainAndY(6, -1, -1, -1)._average_normalized_height_a  # doctest: +NORMALIZE_WHITESPACE
            array([Fraction(1, 7), Fraction(2, 7), Fraction(3, 7), Fraction(4, 7),
                   Fraction(5, 7), Fraction(6, 7)], dtype=object)
        """
        return np.array([Fraction(i + 1, self.a + 1) for i in range(self.a)])

    @property
    def _average_normalized_height_b(self):
        """
        Average normalized height for trunk `b`.

        Returns
        -------
        :class:`~numpy.ndarray`
            Size `b`. Average normalized height for each node.

        Examples
        --------
        Since it does not depend on `a`, it is not necessary to give this parameter:

            >>> ChainAndY(-1, 6, 2, 2)._average_normalized_height_b  # doctest: +NORMALIZE_WHITESPACE
            array([Fraction(1, 11), Fraction(2, 11), Fraction(3, 11), Fraction(4, 11),
                   Fraction(5, 11), Fraction(6, 11)], dtype=object)
        """
        # Note that it would be the same with a chain of c + d above b.
        return np.array([Fraction(i+1, self.b + self.c + self.d + 1) for i in range(self.b)])

    @property
    def _average_height_c(self):
        """
        Average height for branch `c`.

        Returns
        -------
        :class:`~numpy.ndarray`
            Size `c`. Average height for each node.

        Examples
        --------
            >>> ChainAndY(2, 0, 7, 0)._average_height_c  # doctest: +NORMALIZE_WHITESPACE
            array([Fraction(5, 4), Fraction(5, 2), Fraction(15, 4), Fraction(5, 1),
                   Fraction(25, 4), Fraction(15, 2), Fraction(35, 4)], dtype=object)
        """
        if self._cache_average_height_c is None:
            self._cache_average_height_c = _average_height_c(self.a, self.b, self.c, self.d)
        return self._cache_average_height_c

    @property
    def _average_height_d(self):
        """
        Average height for branch `d`.

        Returns
        -------
        :class:`~numpy.ndarray`
            Size `d`. Average height for each node.

        Examples
        --------
            >>> ChainAndY(2, 0, 0, 7)._average_height_d  # doctest: +NORMALIZE_WHITESPACE
            array([Fraction(5, 4), Fraction(5, 2), Fraction(15, 4), Fraction(5, 1),
                   Fraction(25, 4), Fraction(15, 2), Fraction(35, 4)], dtype=object)
        """
        if self._cache_average_height_d is None:
            self._cache_average_height_d = _average_height_c(self.a, self.b, self.d, self.c)
        return self._cache_average_height_d

    @property
    def average_normalized_height(self):
        """
        Average normalized height.

        Returns
        -------
        :class:`~numpy.ndarray`
            Size `c`. Average normalized height for each node.

        Examples
        --------
            >>> ChainAndY(2, 1, 7, 3).average_normalized_height  # doctest: +NORMALIZE_WHITESPACE
            array([Fraction(1, 3), Fraction(2, 3), Fraction(1, 12), Fraction(19, 96),
                   Fraction(5, 16), Fraction(41, 96), Fraction(13, 24),
                   Fraction(21, 32), Fraction(37, 48), Fraction(85, 96),
                   Fraction(5, 16), Fraction(13, 24), Fraction(37, 48)], dtype=object)
        """
        return np.concatenate((
            self._average_normalized_height_a,
            self._average_normalized_height_b,
            self._average_height_c / (self.n_nodes + 1),
            self._average_height_d / (self.n_nodes + 1),
        ))

    @property
    def average_height(self):
        """
        Average height.

        Returns
        -------
        :class:`~numpy.ndarray`
            Size `c`. Average height for each node.

        Examples
        --------
            >>> ChainAndY(2, 1, 7, 3).average_height  # doctest: +NORMALIZE_WHITESPACE
            array([Fraction(14, 3), Fraction(28, 3), Fraction(7, 6),
                   Fraction(133, 48), Fraction(35, 8), Fraction(287, 48),
                   Fraction(91, 12), Fraction(147, 16), Fraction(259, 24),
                   Fraction(595, 48), Fraction(35, 8), Fraction(91, 12),
                   Fraction(259, 24)], dtype=object)
        """
        return np.concatenate((
            self._average_normalized_height_a * (self.n_nodes + 1),
            self._average_normalized_height_b * (self.n_nodes + 1),
            self._average_height_c,
            self._average_height_d,
        ))

    @property
    def _positions_counts_in_extensions_a(self):
        """
        Positions counts in linear extensions for the elements of the chain `a`.

        Returns
        -------
        :class:`~numpy.ndarray`
            Size `a` * `a + b + c + d`. Coefficient (i, r) represents the number of linear extensions of the
            poset where item `i` of the chain is in rank `r`.

        Examples
        --------
            >>> ChainAndY(2, 1, 7, 3)._positions_counts_in_extensions_a  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
            array([[1440, 1320, 1200, 1080,  960,  840,  720,  600,  480,  360,  240,
                     120,    0],
                   [   0,  120,  240,  360,  480,  600,  720,  840,  960, 1080, 1200,
                    1320, 1440]]...)
        """
        result = np.zeros((self.a, self.a + self.b + self.c + self.d), 'int64')
        ways_to_merge_c_and_d = comb(self.c + self.d, self.c)
        for i in range(self.a):  # i: index of element in chain `a`
            smaller_from_a = i
            greater_from_a = self.a - smaller_from_a - 1
            for smaller_from_y in range(self.b + self.c + self.d + 1):
                greater_from_y = self.b + self.c + self.d - smaller_from_y
                final_rank = smaller_from_a + smaller_from_y
                ways_left = comb(smaller_from_a + smaller_from_y, smaller_from_a)
                ways_right = comb(greater_from_a + greater_from_y, greater_from_a)
                ways = ways_left * ways_right * ways_to_merge_c_and_d
                result[i, final_rank] = ways
        return result

    @property
    def _positions_counts_in_extensions_b(self):
        """
        Positions counts in linear extensions for the elements of the trunk `b`.

        Returns
        -------
        :class:`~numpy.ndarray`
            Size `b` * `a + b + c + d`. Coefficient (i, r) represents the number of linear extensions of the
            poset where item `i` of the trunk is in rank `r`.

        Examples
        --------
            >>> ChainAndY(2, 1, 7, 3)._positions_counts_in_extensions_b  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
            array([[7920, 1320,  120,    0,    0,    0,    0,    0,    0,    0,    0,
                       0,    0]]...)
        """
        result = np.zeros((self.b, self.a + self.b + self.c + self.d), 'int64')
        ways_to_merge_c_and_d = comb(self.c + self.d, self.c)
        for i in range(self.b):  # i: index of element in chain `b`
            smaller_from_y = i
            greater_from_y = self.b + self.c + self.d - smaller_from_y - 1
            for smaller_from_a in range(self.a + 1):
                greater_from_a = self.a - smaller_from_a
                final_rank = smaller_from_a + smaller_from_y
                ways_left = comb(smaller_from_a + smaller_from_y, smaller_from_a)
                ways_right = comb(greater_from_a + greater_from_y, greater_from_a)
                ways = ways_left * ways_right * ways_to_merge_c_and_d
                result[i, final_rank] = ways
        return result

    @property
    def _positions_counts_in_extensions_c(self):
        """
        Positions counts in linear extensions for the elements of the branch `c`.

        Returns
        -------
        :class:`~numpy.ndarray`
            Size `c` * `a + b + c + d`. Coefficient (i, r) represents the number of linear extensions of the
            poset where item `i` of the branch is in rank `r`.

        Examples
        --------
            >>> ChainAndY(2, 1, 7, 3)._positions_counts_in_extensions_c  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
            array([[   0, 4620, 2940, 1260,  420,  105,   15,    0,    0,    0,    0,
                       0,    0],
                   [   0,    0, 2520, 3024, 2184, 1134,  414,   84,    0,    0,    0,
                       0,    0],
                   [   0,    0,    0, 1260, 2380, 2555, 1905,  980,  280,    0,    0,
                       0,    0],
                   [   0,    0,    0,    0,  560, 1540, 2340, 2440, 1760,  720,    0,
                       0,    0],
                   [   0,    0,    0,    0,    0,  210,  810, 1710, 2490, 2565, 1575,
                       0,    0],
                   [   0,    0,    0,    0,    0,    0,   60,  320,  940, 1950, 3010,
                    3080,    0],
                   [   0,    0,    0,    0,    0,    0,    0,   10,   74,  309,  959,
                    2464, 5544]]...)
        """
        return _positions_counts_in_extensions_c(self.a, self.b, self.c, self.d)

    @property
    def _positions_counts_in_extensions_d(self):
        """
        Positions counts in linear extensions for the elements of the branch `d`.

        Returns
        -------
        :class:`~numpy.ndarray`
            Size `d` * `a + b + c + d`. Coefficient (i, r) represents the number of linear extensions of the
            poset where item `i` of the branch is in rank `r`.

        Examples
        --------
            >>> ChainAndY(2, 1, 7, 3)._positions_counts_in_extensions_d  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
            array([[   0, 1980, 1980, 1620, 1260,  945,  675,  450,  270,  135,   45,
                       0,    0],
                   [   0,    0,  360,  720, 1000, 1190, 1290, 1300, 1220, 1050,  790,
                     440,    0],
                   [   0,    0,    0,   36,  116,  241,  411,  626,  886, 1191, 1541,
                    1936, 2376]]...)
        """
        return _positions_counts_in_extensions_c(self.a, self.b, self.d, self.c)

    @property
    def positions_counts_in_extensions(self):
        """
        Positions counts in linear extensions for the elements of the poset.

        Returns
        -------
        :class:`~numpy.ndarray`
            Size `a + b + c + d` * `a + b + c + d`. Coefficient (i, r) represents the number of linear extensions of the
            poset where node `i` is in rank `r` (note that items and ranks are numbered from 0, Python-style).

        Examples
        --------
            >>> ChainAndY(2, 1, 7, 3).positions_counts_in_extensions  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
            array([[1440, 1320, 1200, 1080,  960,  840,  720,  600,  480,  360,  240,
                     120,    0],
                   [   0,  120,  240,  360,  480,  600,  720,  840,  960, 1080, 1200,
                    1320, 1440],
                   [7920, 1320,  120,    0,    0,    0,    0,    0,    0,    0,    0,
                       0,    0],
                   [   0, 4620, 2940, 1260,  420,  105,   15,    0,    0,    0,    0,
                       0,    0],
                   [   0,    0, 2520, 3024, 2184, 1134,  414,   84,    0,    0,    0,
                       0,    0],
                   [   0,    0,    0, 1260, 2380, 2555, 1905,  980,  280,    0,    0,
                       0,    0],
                   [   0,    0,    0,    0,  560, 1540, 2340, 2440, 1760,  720,    0,
                       0,    0],
                   [   0,    0,    0,    0,    0,  210,  810, 1710, 2490, 2565, 1575,
                       0,    0],
                   [   0,    0,    0,    0,    0,    0,   60,  320,  940, 1950, 3010,
                    3080,    0],
                   [   0,    0,    0,    0,    0,    0,    0,   10,   74,  309,  959,
                    2464, 5544],
                   [   0, 1980, 1980, 1620, 1260,  945,  675,  450,  270,  135,   45,
                       0,    0],
                   [   0,    0,  360,  720, 1000, 1190, 1290, 1300, 1220, 1050,  790,
                     440,    0],
                   [   0,    0,    0,   36,  116,  241,  411,  626,  886, 1191, 1541,
                    1936, 2376]]...)
        """
        if self._cache_positions_counts_in_extensions is None:
            self._cache_positions_counts_in_extensions = np.vstack((
                self._positions_counts_in_extensions_a,
                self._positions_counts_in_extensions_b,
                self._positions_counts_in_extensions_c,
                self._positions_counts_in_extensions_d
            ))
        return self._cache_positions_counts_in_extensions

    @property
    def median_height(self):
        """
        Median height.

        Returns
        -------
        :class:`~numpy.ndarray`
            Size `a + b + c + d`. Median height for each node in the linear extensions.

        Examples
        --------
            >>> ChainAndY(2, 1, 7, 3).median_height  # doctest: +NORMALIZE_WHITESPACE
            array([ 3.,  9.,  0.,  2.,  3.,  5.,  7.,  8., 10., 12.,  3.,  7., 10.])
        """
        m = self.positions_counts_in_extensions
        m_cumsum = m.cumsum(axis=1)
        nb_linear_extensions = self.nb_linear_extensions
        median_low = np.argmax(m_cumsum >= nb_linear_extensions / 2, axis=1)
        median_high = np.argmax(m_cumsum > nb_linear_extensions / 2, axis=1)
        return (median_low + median_high) / 2

    @property
    def spearman_costs(self):
        """
        Spearman costs.

        Returns
        -------
        :class:`~numpy.ndarray`
            Size `a + b + c + d` * `a + b + c + d`. Coefficient (i, r) represents the contribution of item `i` to the
            Spearman distance if it is placed in position `r` in the estimate ranking.

        Examples
        --------
            >>> ChainAndY(2, 1, 7, 3).spearman_costs  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
            array([[ 34320,  27840,  24000,  22560,  23280,  25920,  30240,  36000,
                     42960,  50880,  59520,  68640,  78000],
                   [ 78000,  68640,  59520,  50880,  42960,  36000,  30240,  25920,
                     23280,  22560,  24000,  27840,  34320],
                   [  1560,   8040,  17160,  26520,  35880,  45240,  54600,  63960,
                     73320,  82680,  92040, 101400, 110760],
                   [ 16575,   7215,   7095,  12855,  21135,  30255,  39585,  48945,
                     58305,  67665,  77025,  86385,  95745],
                   [ 31590,  22230,  12870,   8550,  10278,  16374,  24738,  33930,
                     43290,  52650,  62010,  71370,  80730],
                   [ 46605,  37245,  27885,  18525,  11685,   9605,  12635,  19475,
                     28275,  37635,  46995,  56355,  65715],
                   [ 61620,  52260,  42900,  33540,  24180,  15940,  10780,  10300,
                     14700,  22620,  31980,  41340,  50700],
                   [ 76635,  67275,  57915,  48555,  39195,  29835,  20895,  13575,
                      9675,  10755,  16965,  26325,  35685],
                   [ 91650,  82290,  72930,  63570,  54210,  44850,  35490,  26250,
                     17650,  10930,   8110,  11310,  20670],
                   [106665,  97305,  87945,  78585,  69225,  59865,  50505,  41145,
                     31805,  22613,  14039,   7383,   5655],
                   [ 31590,  22230,  16830,  15390,  17190,  21510,  27720,  35280,
                     43740,  52740,  62010,  71370,  80730],
                   [ 61620,  52260,  42900,  34260,  27060,  21860,  19040,  18800,
                     21160,  25960,  32860,  41340,  50700],
                   [ 91650,  82290,  72930,  63570,  54282,  45226,  36652,  28900,
                     22400,  17672,  15326,  16062,  20670]]...)
        """
        distances = np.array([
            [abs(i - j) for i in range(self.n_nodes)]
            for j in range(self.n_nodes)
        ])
        return self.positions_counts_in_extensions @ distances

    @property
    def order_spearman_optimal(self):
        """
        Order of the nodes optimizing the expected Spearman distance.

        Returns
        -------
        :class:`~numpy.ndarray`
            The candidates, in their optimal order for expected Spearman distance.

        Examples
        --------
            >>> ChainAndY(0, 1, 2, 7).order_spearman_optimal  # doctest: +ELLIPSIS
            array([0, 3, 4, 1, 5, 6, 7, 2, 8, 9]...)
        """
        _, col_ind = linear_sum_assignment(self.spearman_costs.T)
        return col_ind

    @property
    def profile_linear_extensions(self):
        """
        Profile consisting of all linear extensions.

        Returns
        -------
        :class:`~numpy.ndarray`
            One row for each linear extension, `a + b + c + d` columns. Each coefficient is the label of a node.

        Examples
        --------
            >>> ChainAndY(1, 2, 1, 2).profile_linear_extensions
            array([[0, 1, 2, 3, 4, 5],
                   [1, 0, 2, 3, 4, 5],
                   [1, 2, 0, 3, 4, 5],
                   [1, 2, 3, 0, 4, 5],
                   [1, 2, 3, 4, 0, 5],
                   [1, 2, 3, 4, 5, 0],
                   [0, 1, 2, 4, 3, 5],
                   [1, 0, 2, 4, 3, 5],
                   [1, 2, 0, 4, 3, 5],
                   [1, 2, 4, 0, 3, 5],
                   [1, 2, 4, 3, 0, 5],
                   [1, 2, 4, 3, 5, 0],
                   [0, 1, 2, 4, 5, 3],
                   [1, 0, 2, 4, 5, 3],
                   [1, 2, 0, 4, 5, 3],
                   [1, 2, 4, 0, 5, 3],
                   [1, 2, 4, 5, 0, 3],
                   [1, 2, 4, 5, 3, 0]])
        """
        if self._cache_profile_linear_extensions is None:
            profile = []
            elements_a = np.array(range(self.a), dtype=int)
            elements_b = np.array(range(self.a, self.a + self.b), dtype=int)
            elements_c = np.array(range(self.a + self.b, self.a + self.b + self.c), dtype=int)
            elements_d = np.array(range(self.a + self.b + self.c, self.a + self.b + self.c + self.d), dtype=int)
            for extension_c_d in linear_extensions(chain_1=elements_c, chain_2=elements_d):
                extension_b_c_d = np.concatenate((elements_b, extension_c_d))
                for extension in linear_extensions(chain_1=elements_a, chain_2=extension_b_c_d):
                    profile.append(extension)
            self._cache_profile_linear_extensions = np.array(profile)
        return self._cache_profile_linear_extensions

    @property
    def profile_linear_extensions_svvamp(self):
        """
        Profile consisting of all linear extensions, as a svvamp profile.

        Returns
        -------
        :class:`~svvamp.Profile`
            Each voter represents a linear extension of the poset.

        Examples
        --------
        The attribute `preferences_rk` of the svvamp profile is equal to the attribute
        :meth:`profile_linear_extensions` of the poset:

            >>> ChainAndY(1, 2, 1, 2).profile_linear_extensions_svvamp.preferences_rk
            array([[0, 1, 2, 3, 4, 5],
                   [1, 0, 2, 3, 4, 5],
                   [1, 2, 0, 3, 4, 5],
                   [1, 2, 3, 0, 4, 5],
                   [1, 2, 3, 4, 0, 5],
                   [1, 2, 3, 4, 5, 0],
                   [0, 1, 2, 4, 3, 5],
                   [1, 0, 2, 4, 3, 5],
                   [1, 2, 0, 4, 3, 5],
                   [1, 2, 4, 0, 3, 5],
                   [1, 2, 4, 3, 0, 5],
                   [1, 2, 4, 3, 5, 0],
                   [0, 1, 2, 4, 5, 3],
                   [1, 0, 2, 4, 5, 3],
                   [1, 2, 0, 4, 5, 3],
                   [1, 2, 4, 0, 5, 3],
                   [1, 2, 4, 5, 0, 3],
                   [1, 2, 4, 5, 3, 0]])
        """
        if self._cache_profile_linear_extensions_svvamp is None:
            self._cache_profile_linear_extensions_svvamp = svvamp.Profile(preferences_rk=self.profile_linear_extensions)
        return self._cache_profile_linear_extensions_svvamp

    @property
    def order_kemeny(self):
        """
        Kemeny order of the nodes in the profile of linear extensions.

        Returns
        -------
        :class:`~numpy.ndarray`
            The candidates, in their Kemeny order.

        Examples
        --------
            >>> ChainAndY(0, 1, 2, 7).order_kemeny
            array([0, 3, 4, 1, 5, 6, 7, 2, 8, 9])
        """
        election = svvamp.RuleKemeny()(self.profile_linear_extensions_svvamp)
        return election.candidates_by_scores_best_to_worst_

    def kemeny_score(self, order):
        """
        Kemeny score of an order over the candidates.

        Parameters
        ----------
        order: :class:`list`
            An order over the candidates.

        Returns
        -------
        float
            The Kemeny score (average kendall-tau distance with a linear extension of the poset).

        Examples
        --------
            >>> poset = ChainAndY(2, 2, 1, 3)
            >>> poset.kemeny_score([0, 4, 7, 2, 6, 3, 5, 1])
            13.5
        """
        profile = self.profile_linear_extensions_svvamp
        n_extensions = profile.n_v
        return np.sum(np.tril(profile.matrix_duels_rk[order, :][:, order], -1)) / n_extensions

    def spearman_score(self, order):
        """
        Spearman score of an order over the candidates.

        Parameters
        ----------
        order: :class:`list`
            An order over the candidates.

        Returns
        -------
        float
            The Spearman score (average Spearman distance with a linear extension of the poset).

        Examples
        --------
            >>> poset = ChainAndY(2, 2, 1, 3)
            >>> poset.spearman_score([0, 4, 7, 2, 6, 3, 5, 1])
            22.142857142857142
        """
        spearman_costs = self.spearman_costs
        total = 0
        for rank, item in enumerate(order):
            total += (spearman_costs[item, rank] / self.nb_linear_extensions)
        return total

    @property
    def order_delta(self):
        """
        Order of the nodes, according to estimator delta.

        Returns
        -------
        :class:`~numpy.ndarray`
            Estimated order of the nodes.

        Examples
        --------
            >>> ChainAndY(2, 2, 1, 3).order_delta  # doctest: +ELLIPSIS
            array([2, 3, 0, 5, 1, 4, 6, 7]...)
        """
        return np.argsort(self.delta)

    @property
    def order_rho(self):
        """
        Order of the nodes, according to estimator rho.

        Returns
        -------
        :class:`~numpy.ndarray`
            Estimated order of the nodes.

        Examples
        --------
            >>> ChainAndY(2, 2, 1, 3).order_rho  # doctest: +ELLIPSIS
            array([2, 3, 0, 5, 1, 6, 4, 7]...)
        """
        return np.argsort(self.rho)

    @property
    def order_average_height(self):
        """
        Order of the nodes, according to average height.

        Returns
        -------
        :class:`~numpy.ndarray`
            Estimated order of the nodes.

        Examples
        --------
            >>> ChainAndY(2, 2, 1, 3).order_average_height  # doctest: +ELLIPSIS
            array([2, 3, 0, 5, 4, 6, 1, 7]...)
        """
        return np.argsort(self.average_height)

    @property
    def order_median_height(self):
        """
        Order of the nodes, according to median height.

        Returns
        -------
        :class:`~numpy.ndarray`
            Estimated order of the nodes.

        Examples
        --------
            >>> ChainAndY(2, 2, 1, 3).order_median_height  # doctest: +ELLIPSIS
            array([2, 3, 0, 5, 1, 4, 6, 7]...)
        """
        return np.argsort(self.median_height)

    def _print_latex_delta_rho_m(self):
        """
        Print the LaTeX code for the article.
        """
        print(r"""\begin{tikzpicture}[scale=.45, transform shape]
\footnotesize
\foreach \label/\desc/\anc/\x/\m/\mn in {% \m: median height in [1, n] ; \mn: median height normalized by n+1""")
        nb_descendants = self.nb_descendants
        nb_ancestors = self.nb_ancestors
        median_height = self.median_height
        median_height_normalized = (self.median_height + 1) / (self.n_nodes + 1)
        lst = ", \n".join([
            f"  {string.ascii_lowercase[i]}/{nb_descendants[i]}/{nb_ancestors[i]}/{x}/{int(median_height[i] + 1)}/"
            + f"{median_height_normalized[i]}"
            for i, x in zip(range(self.n_nodes), [-2] * self.a + [0] * self.b + [-1] * self.c + [1] * self.d)
        ])
        print(lst)
        print(r"""}
{
  \pgfmathtruncatemacro{\to}{\desc + \anc}
  \pgfmathtruncatemacro{\delta}{\desc - \anc}
  \node[obj,inner sep=1pt,minimum size=1pt] (d\label) at (1.15*\x, 7.01+.624*13/15*\delta) {\Large\strut$\label$};
  \node[right = -.5mm of d\label] {\Large$\delta$};
  \node[obj,inner sep=1pt,minimum size=1pt] (r\label) at (7+1.4*\x, 16.5*\desc/\to - 1.5) {\Large\strut$\label$};
  \node[right = -.5mm of r\label] {\Large$\frac{\desc}{\to} $};
  \node[obj,inner sep=1pt,minimum size=1pt] (h\label) at (14+1.6*\x, 16.5*\mn - 1.5) {\Large\strut$\label$};
  \node[right = -.5mm of h\label] {\Large $\m$};
}
\node at (0, 0) {\LARGE(a) $\Delta$ scores.};
\node at (7, 0) {\LARGE(b) $\rho$ scores.};
\node at (14, 0) {\LARGE(c) $m$ scores.};""")
        labels_a = string.ascii_lowercase[0:self.a]
        labels_b = string.ascii_lowercase[self.a:self.a + self.b]
        labels_c = string.ascii_lowercase[self.a + self.b:self.a + self.b + self.c]
        labels_d = string.ascii_lowercase[self.a + self.b + self.c:self.a + self.b + self.c + self.d]
        edges = ", ".join(
            [f"{x}/{y}" for x, y in zip(labels_a[:-1], labels_a[1:])]
            + [f"{x}/{y}" for x, y in zip(labels_b[:-1], labels_b[1:])]
            + [f"{labels_b[-1]}/{labels_c[0]}"]
            + [f"{x}/{y}" for x, y in zip(labels_c[:-1], labels_c[1:])]
            + [f"{labels_b[-1]}/{labels_d[0]}"]
            + [f"{x}/{y}" for x, y in zip(labels_d[:-1], labels_d[1:])]
        )
        print(r"\foreach \i/\j in {" + f"{edges}" + r"}{%")
        print(r"""  \draw[<-] (d\i) -- (d\j);
  \draw[<-] (r\i) -- (r\j);
  \draw[<-] (h\i) -- (h\j);
}
\end{tikzpicture}""")


def _average_height_c(a, b, c, d):
    """
    Average height for branch `c`.

    Parameters
    ----------
    a: int
        Number of nodes in the isolated chain.
    b: int
        Number of nodes in the trunk of the Y.
    c, d: int
        Number of nodes in each branch of the Y.

    Returns
    -------
    :class:`~numpy.ndarray`
        Size `c`. Average height for each node.

    Examples
    --------
        >>> _average_height_c(2, 0, 7, 0)  # doctest: +NORMALIZE_WHITESPACE
        array([Fraction(5, 4), Fraction(5, 2), Fraction(15, 4), Fraction(5, 1),
               Fraction(25, 4), Fraction(15, 2), Fraction(35, 4)], dtype=object)
    """
    result = np.zeros(c, dtype='int64')
    for k in range(c):
        # Elements of c are: 0 ... (k-1) k (k+1) .. (c-1)
        smaller_from_c = k
        greater_from_c = c - 1 - k
        for smaller_from_a in range(a + 1):
            greater_from_a = a - smaller_from_a
            for smaller_from_d in range(d + 1):
                greater_from_d = d - smaller_from_d
                height = smaller_from_a + b + smaller_from_c + smaller_from_d + 1
                n_lin_ext = (
                    ChainAndY(smaller_from_a, b, smaller_from_c, smaller_from_d).nb_linear_extensions
                    * ChainAndY(greater_from_a, 0, greater_from_c, greater_from_d).nb_linear_extensions
                )
                result[k] += n_lin_ext * height
    nb_lin_ext = ChainAndY(a, b, c, d).nb_linear_extensions
    return np.array([Fraction(r, nb_lin_ext) for r in result])


def _positions_counts_in_extensions_c(a, b, c, d):
    """
    Positions counts in linear extensions for the elements of the branch `c`.

    Parameters
    ----------
    a: int
        Number of nodes in the isolated chain.
    b: int
        Number of nodes in the trunk of the Y.
    c, d: int
        Number of nodes in each branch of the Y.

    Returns
    -------
    :class:`~numpy.ndarray`
        Size `c` * `a + b + c + d`. Coefficient (i, r) represents the number of linear extensions of the
        poset where item `i` of the branch is in rank `r`.

    Examples
    --------
        >>> _positions_counts_in_extensions_c(2, 0, 7, 0)  # doctest: +NORMALIZE_WHITESPACE +ELLIPSIS
        array([[28,  7,  1,  0,  0,  0,  0,  0,  0],
               [ 0, 21, 12,  3,  0,  0,  0,  0,  0],
               [ 0,  0, 15, 15,  6,  0,  0,  0,  0],
               [ 0,  0,  0, 10, 16, 10,  0,  0,  0],
               [ 0,  0,  0,  0,  6, 15, 15,  0,  0],
               [ 0,  0,  0,  0,  0,  3, 12, 21,  0],
               [ 0,  0,  0,  0,  0,  0,  1,  7, 28]]...)
    """
    result = np.zeros((c, a + b + c + d), 'int64')
    for k in range(c):
        # Elements of c are: 0 ... (k-1) k (k+1) .. (c-1)
        smaller_from_c = k
        greater_from_c = c - 1 - k
        for smaller_from_a in range(a + 1):
            greater_from_a = a - smaller_from_a
            for smaller_from_d in range(d + 1):
                greater_from_d = d - smaller_from_d
                height = smaller_from_a + b + smaller_from_c + smaller_from_d
                n_lin_ext = (
                    ChainAndY(smaller_from_a, b, smaller_from_c, smaller_from_d).nb_linear_extensions
                    * ChainAndY(greater_from_a, 0, greater_from_c, greater_from_d).nb_linear_extensions
                )
                result[k, height] += n_lin_ext
    return result


def linear_extensions(chain_1, chain_2):
    """
    Linear extensions for two independent chains (separate connected components).

    Parameters
    ----------
    chain_1: class:`list`
    chain_2: class:`list`

    Yields
    ------
    extension: class:`list`
        A linear extension of the poset consisting of the two given chains.

    Examples
    --------
        >>> for linear_extension in linear_extensions([1, 2, 3], [42, 52]):
        ...     print(linear_extension)
        [ 1  2  3 42 52]
        [ 1  2 42  3 52]
        [ 1  2 42 52  3]
        [ 1 42  2  3 52]
        [ 1 42  2 52  3]
        [ 1 42 52  2  3]
        [42  1  2  3 52]
        [42  1  2 52  3]
        [42  1 52  2  3]
        [42 52  1  2  3]

        >>> for linear_extension in linear_extensions([1, 2, 3], []):
        ...     print(linear_extension)
        [1 2 3]
    """
    #
    x = len(chain_1)
    y = len(chain_2)
    if x == 0 or y == 0:
        yield np.array(np.concatenate((chain_1, chain_2)), dtype=int)
    else:
        for places_x in combinations(range(x + y), x):
            extension = np.zeros(x + y, dtype=int)
            extension[np.array(places_x)] = chain_1
            mask_y = np.ones(x + y, dtype=bool)
            mask_y[np.array(places_x)] = False
            extension[mask_y] = chain_2
            yield extension
