import numpy as np


def multi_merge(xs, list_indices, lt=None):
    """
    Merge consecutive sorted portions of a list, two by two, in alternance.

    Assume that xs[i:j] and xs[j:k] are already sorted, and merge-sort them.

    Parameters
    ----------
    xs: :class:`list`
        Values to sort.
    list_indices: :class:`~numpy.ndarray`
        Indices of the portions (cf. example below).
    lt: callable
        lt(x, y) is the test used to determine whether element x is lower than y.
        Default: operator "<".

    Examples
    --------
        >>> my_xs = [2, 5, 0, 1, 7, 8, 3, 4, 6]
        >>> my_list_indices = np.array([0, 2, 4, 6, 9])
        >>> multi_merge(my_xs, my_list_indices)
        >>> my_xs
        [0, 1, 2, 5, 3, 4, 6, 7, 8]
    """
    if lt is None:
        def lt(x, y):
            return x < y
    begins_chains = list_indices.copy()
    finished = False
    while not finished:
        finished = True
        for i in range(0, len(list_indices) - 1, 2):
            begin_left = begins_chains[i]
            begin_right = begins_chains[i + 1]
            k = list_indices[i + 2]
            if begin_left < begin_right < k:
                finished = False
                if lt(xs[begin_left], xs[begin_right]):
                    begins_chains[i] += 1
                else:
                    item_to_insert = xs[begin_right]
                    xs[begin_left+1:begin_right+1] = xs[begin_left:begin_right]
                    xs[begin_left] = item_to_insert
                    begins_chains[i] += 1
                    begins_chains[i + 1] += 1
