def merge(xs, i, j, k, lt=None):
    """
    Merge two consecutive sorted portions of a list.

    Assume that xs[i:j] and xs[j:k] are already sorted, and merge-sort them.

    Parameters
    ----------
    xs: :class:`list`
        Values to sort.
    i: :class:`int`
        Beginning of the left portion.
    j: :class:`int`
        Beginning of the right portion.
    k: :class:`int`
        End (excluded) of the right portion.
    lt: callable
        lt(x, y) is the test used to determine whether element x is lower than y.
        Default: operator "<".

    Examples
    --------
        >>> my_xs = [0, 1, 4, 6, 7, 2, 3, 5, 8]
        >>> merge(my_xs, i=0, j=5, k=9)
        >>> my_xs
        [0, 1, 2, 3, 4, 5, 6, 7, 8]
    """
    if lt is None:
        def lt(x, y):
            return x < y
    begin_left = i
    begin_right = j
    while begin_left < begin_right < k:
        if lt(xs[begin_left], xs[begin_right]):
            begin_left += 1
        else:
            item_to_insert = xs[begin_right]
            xs[begin_left+1:begin_right+1] = xs[begin_left:begin_right]
            xs[begin_left] = item_to_insert
            begin_left += 1
            begin_right += 1
