import numpy as np


def partition(xs, i, j, lt):
    """
    Move the pivot element to the right place and return its position (for quicksort or quickselect).

    Parameters
    ----------
    xs: :class:`~numpy.ndarray`
        Array to sort.
    i: :class:`int`
        Index of the left boundary.
    j: :class:`int`
        Index of the right boundary.
    lt: callable
        lt(x, y) is the test used to determine whether element x is lower than y.

    Returns
    -------
    :class:`int`
        Pivot index.

    Examples
    --------
        >>> my_xs = np.array([4, 1, 7, 6, 0, 8, 2, 3, 5])
        >>> partition(my_xs, i=0, j=len(my_xs) - 1, lt=lambda x, y: x < y)
        4
        >>> my_xs
        array([1, 0, 2, 3, 4, 8, 6, 7, 5])
    """
    pivot_value = xs[i]
    pivot_index = i
    for k in range(i + 1, j + 1):
        if lt(xs[k], pivot_value):
            if k > pivot_index + 1:
                xs[pivot_index], xs[pivot_index + 1], xs[k] = xs[k], pivot_value, xs[pivot_index + 1]
            else:
                xs[pivot_index], xs[k] = xs[k], pivot_value
            pivot_index += 1
    return pivot_index
