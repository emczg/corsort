from numba import njit
import numpy as np



@njit
def quicksort(xs, i = 0, j = None, nc=0):
    """
    Inspired by https://codereview.stackexchange.com/questions/272639/in-place-quicksort-algorithm-in-python

    Parameters
    ----------
    xs: :class:`~numpy.ndarray`
        Array to sort.
    i: :class:`int`
        Left boundary
    j: :class:`int`
        Right boundary
    nc: :class:`int`
        Comparison counter

    Returns
    -------
    :class:`int`
        Number of comparison

    Examples
    --------

    >>> np.random.seed(42)
    >>> quicksort(np.random.permutation(100))
    635
    """
    # Set optional arguments.
    if j is None:
        j = len(xs) - 1

    # Base case: do nothing if indexes have met or crossed.
    if not i < j:
        return nc

    # Partition the sequence to enforce the quicksort invariant:
    # "small values" < pivot value <= "large values". The function
    # returns the index of the pivot value.
    pi, nc = partition(xs, i, j, nc)

    # Sort left side and right side.
    nc = quicksort(xs, i=i, j=pi - 1, nc=nc)
    nc = quicksort(xs, i=pi + 1, j=j, nc=nc)
    return nc

@njit
def partition(xs, i, j, nc):
    """

    Parameters
    ----------
    xs: :class:`~numpy.ndarray`
        Array to sort.
    i: :class:`int`
        Left boundary
    j: :class:`int`
        Right boundary
    nc: :class:`int`
        Comparison counter

    Returns
    -------
    :class:`int`
        Pivot index
    :class:`int`
        Comparison counter
    """
    # Get the pivot value and initialize the partition boundary.
    pval = xs[i]
    pb = i + 1

    # Examine all values other than the pivot, swapping to enforce the
    # invariant. Every swap moves an observed "small" value to the left of the
    # boundary. "Large" values are left alone since they are already to the
    # right of the boundary.
    for k in range(pb, j + 1):
        nc += 1
        if xs[k] < pval:
            swap(xs, k, pb)
            pb += 1
        # Possible distance computation here

    # Put pivot value between the two sides of the partition,
    # and return that location.
    swap(xs, i, pb - 1)
    return pb - 1, nc

@njit
def swap(xs, i, j):
    """
    Inplace swap.

    Parameters
    ----------
    xs: :class:`~numpy.ndarray`
        Array to sort.
    i: :class:`int`
        First element to swap.
    j: :class:`int`
        Second element to swap.

    Returns
    -------
    None
    """
    xs[i], xs[j] = xs[j], xs[i]
