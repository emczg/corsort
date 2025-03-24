import numpy as np
from corsort.sort import Sort
from corsort.distance_to_sorted_array import distance_to_sorted_array

from numba import njit



@njit
def jit_update_scores(l, lr, r, rb, b, indices, scores):
    d = rb + 1
    dl = lr - l + 1
    dr = rb - r + 1
    bb = b - rb
    # print(d, dl, dr, b)
    for i, k in enumerate(indices[rb:b]):
        scores[k] = (i + 1) / d
    for i, k in enumerate(indices[l:lr]):
        scores[k] = (bb * dl + (d - bb) * (i + 1)) / (d * dl)
    for i, k in enumerate(indices[r:rb]):
        scores[k] = (bb * dr + (d - bb) * (i + 1)) / (d * dr)


def aux_split(lot):
    return [(ss, (ss + ee) // 2, ee) for ss, ee in
            [k for s, m, e in lot for k in [(s, m), (m, e)]] if ee - ss > 1]


def split(n):
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
        self.l = 0
        self.lr = len(left)
        self.r = self.lr
        self.rb = self.lr + len(right)
        self.b = self.rb
        self.e = 2 * self.b
        self.indices = np.zeros(self.e, dtype=int)
        self.indices[self.l:self.lr] = left
        self.indices[self.r:self.rb] = right
        self.n_ = len(left) + len(right)
        self.scores = {**{i: 0.5 for i in left}, **{j: 0.5 for j in right}}

    def __len__(self):
        return self.n_

    def process(self, lt):
        if self.done:
            return False
        i, j = self.indices[self.l], self.indices[self.r]
        rvalue = True
        if lt(i, j):
            self.indices[self.b] = i
            self.l += 1
            self.b += 1
            if self.l == self.lr:
                self.indices[(self.b):] = self.indices[self.r:self.rb]
                self.b = self.e
                self.r = self.rb
                rvalue = False
        else:
            self.indices[self.b] = j
            self.r += 1
            self.b += 1
            if self.r == self.rb:
                self.indices[(self.b):] = self.indices[self.l:self.lr]
                self.b = self.e
                self.l = self.lr
                rvalue = False
        self.update_scores()
        return rvalue

    def __repr__(self):
        return f"Y(left={[i for i in self.indices[self.l:self.lr]]}, right={[i for i in self.indices[self.r:self.rb]]}, bottom={[i for i in self.indices[self.rb:self.b]]})"

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
        # print(d, dl, dr, b)
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
