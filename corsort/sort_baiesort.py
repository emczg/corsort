import numpy as np
from corsort.sort import Sort
from corsort.distance_to_sorted_array import distance_to_sorted_array


def aux_split(lot):
    """
    Compute the next layer of a tree merge.

    Parameters
    ----------
    lot: :class:`list` of :class:`tuple`
        A layer of a tree merge.

    Returns
    -------
    :class:`list` of :class:`tuple`
        The next layer of a tree merge.
    """
    return [(ss, (ss + ee) // 2, ee) for ss, ee in
            [k for s, m, e in lot for k in [(s, m), (m, e)]] if ee - ss > 1]


def split(n):
    """
    Parameters
    ----------
    n: :class:`int`
        Size of the list to sort.

    Returns
    -------
    :class:`list` of :class:`list` of :class:`tuple`
        The merges to perform by increasing layer. Each merge is a :class:`tuple` `(s, m, e)`, meaning that the left
        part of the merge are indices `s:m` and the right part of the merge are indices `m:e`.
        The first element of the list has always one unique element, the final merge (root of the merge tree).
        The last element of the list indicates the lowest merges (2-merges at the bottom of the tree).

    Examples
    --------

    With 2 elements, you just merge them.

    >>> split(2)
    [[(0, 1, 2)]]

    With 10 elements, it gets a bit more complicated.

    >>> split(10)
    [[(0, 5, 10)], [(0, 2, 5), (5, 7, 10)], [(0, 1, 2), (2, 3, 5), (5, 6, 7), (7, 8, 10)], [(3, 4, 5), (8, 9, 10)]]

    Interpretation:

    * In the end (layer 0), you will merge sorted elements of indices 0 to 4 with sorted elements of indices 5 to 9;
    * Layer 1: merge [0, 1] with [2, 3, 4], merge [5, 6] with [7, 8, 9];
    * Layer 2: [0] and [1], [2] and [3, 4], [5] and [6], [7] and [8, 9];
    * Layer 3: [3] and [4], [8] and [9].
    """
    if n < 2:
        return []
    else:
        start = [(0, n // 2, n)]
        res = [start]
        next_layer = aux_split(start)
        while next_layer:
            res.append(next_layer)
            next_layer = aux_split(res[-1])
        return res


class Y:
    def __init__(self, left, right):
        """
        The `Y` is in charge of zipping `left` and `right`, two sorted lists.

        Parameters
        ----------
        left: :class:`list`
            List (sorted).
        right: :class:`list`
            List (sorted).

        Examples
        --------

        >>> left = [0, 1, 4, 6, 8]
        >>> right = [2, 3, 5, 7, 9]
        >>> y = Y(left, right)
        >>> y
        Y(left=[0, 1, 4, 6, 8], right=[2, 3, 5, 7, 9], bottom=[])
        >>> lt = lambda i, j: i < j
        >>> for _  in range(3):
        ...     c = y.process(lt)
        >>> y
        Y(left=[4, 6, 8], right=[3, 5, 7, 9], bottom=[0, 1, 2])
        >>> while y.process(lt):
        ...     pass
        >>> y
        Y(left=[], right=[], bottom=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        """
        self.l = 0  # Current start of left
        self.lr = len(left)  # end of left
        self.r = self.lr  # Current start of right
        self.rb = self.lr + len(right)  # end of right
        self.b = self.rb  # Current input of bottom
        self.e = 2 * self.b  # end of bottom
        self.indices = np.zeros(self.e, dtype=int)
        self.indices[self.l:self.lr] = left
        self.indices[self.r:self.rb] = right
        self.n_ = len(left) + len(right)
        self.scores = {**{i: 0.5 for i in left}, **{j: 0.5 for j in right}}

    def __len__(self):
        return self.n_

    def process(self, lt):
        """
        Performs one step of the merge.

        Parameters
        ----------
        lt: callable
            Less than function.

        Returns
        -------
        :class:`bool`
            True if there are still some merges to perform.
        """
        if self.done:
            return False
        i, j = self.indices[self.l], self.indices[self.r]
        rvalue = True
        k, choosen_s, choosen_e, other_s, other_e = (i, 'l', 'lr', 'r', 'rb') if lt(i, j) else (j, 'r', 'rb', 'l', 'lr')
        self.indices[self.b] = k
        setattr(self, choosen_s, getattr(self, choosen_s) + 1)
        self.b += 1
        if getattr(self, choosen_s) == getattr(self, choosen_e):
            self.indices[self.b:] = self.indices[getattr(self, other_s):getattr(self, other_e)]
            self.b = self.e
            setattr(self, other_s, getattr(self, other_e))
            rvalue = False
        # if lt(i, j):
        #     self.indices[self.b] = i
        #     self.l += 1
        #     self.b += 1
        #     if self.l == self.lr:
        #         self.indices[(self.b):] = self.indices[self.r:self.rb]
        #         self.b = self.e
        #         self.r = self.rb
        #         rvalue = False
        # else:
        #     self.indices[self.b] = j
        #     self.r += 1
        #     self.b += 1
        #     if self.r == self.rb:
        #         self.indices[(self.b):] = self.indices[self.l:self.lr]
        #         self.b = self.e
        #         self.l = self.lr
        #         rvalue = False
        self.update_scores()
        return rvalue

    def __repr__(self):
        return (f"Y(left={[i for i in self.indices[self.l:self.lr]]}, "
                f"right={[i for i in self.indices[self.r:self.rb]]}, "
                f"bottom={[i for i in self.indices[self.rb:self.b]]})")

    @property
    def done(self):
        return self.b == self.e

    def output(self):
        return self.indices[self.rb:self.b]

    def update_scores(self):
        d = self.rb + 1
        dl = self.lr - self.l + 1
        dr = self.rb - self.r + 1
        b = self.b - self.rb
        for i, k in enumerate(self.indices[self.rb:self.b]):
            self.scores[k] = (i + 1) / d
        for i, k in enumerate(self.indices[self.l:self.lr]):
            self.scores[k] = (b * dl + (d - b) * (i + 1)) / (d * dl)
        for i, k in enumerate(self.indices[self.r:self.rb]):
            self.scores[k] = (b * dr + (d - b) * (i + 1)) / (d * dr)


def yfy(s, m, e, ys):
    left = np.arange(s, m) if m - s < 2 else ys[s].output()
    right = np.arange(m, e) if e - m < 2 else ys[m].output()
    return Y(left, right)


class SortBaie(Sort):
    """
    Merge sort, Baie version (BFS + dedicated scorer).

    Examples
    --------
    >>> baie_sort = SortBaie(compute_history=True)
    >>> my_xs = np.array([4, 1, 7, 6, 0, 8, 2, 3, 5])
    >>> baie_sort(my_xs).n_comparisons_
    19
    >>> baie_sort.history_comparisons_  # doctest: +NORMALIZE_WHITESPACE
    [(7, 8), (1, 0), (3, 2), (4, 5), (6, 7), (1, 3), (4, 6), (0, 3), (6, 5),
    (7, 5), (8, 5), (4, 1), (1, 6), (6, 0), (7, 0), (0, 8), (8, 3), (3, 5), (2, 5)]
    >>> baie_sort.history_distances_
    [30, 28, 28, 22, 16, 16, 14, 14, 10, 8, 8, 4, 4, 4, 4, 2, 2, 2, 0, 0]
    >>> baie_sort.sorted_list_
    array([0, 1, 2, 3, 4, 5, 6, 7, 8])
    """
    __name__ = 'baie_sort'

    def __init__(self, compute_history=False):
        super().__init__(compute_history=compute_history)
        self.ys = None
        self.scores = None

    def _initialize_algo_aux(self):
        pass

    def _call_aux(self):
        lt = self.test_i_lt_j
        layers = split(self.n_)
        self.scores = np.ones(self.n_)/2
        self.ys = dict()
        for layer in layers[::-1]:
            self.ys = {s: yfy(s, m, e, self.ys) for s, m, e in layer}
            # print(layer)
            doit = True
            while doit:
                doit = False
                for y in self.ys.values():
                    doit = y.process(lt) or doit
                    for i, s in y.scores.items():
                        self.scores[i] = s
            # print(ys)

    def distance_to_sorted_array(self):
        return distance_to_sorted_array(self.perm_[self.sorted_indices_])

    @property
    def sorted_indices_(self):
        if self.scores is None:
            return np.arange(self.n_)
        return np.argsort(self.scores)

    @property
    def sorted_list_(self):
        return self.perm_[self.sorted_indices_]
