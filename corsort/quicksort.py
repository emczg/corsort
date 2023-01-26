import numpy as np
from corsort.distance_to_sorted_array import distance_to_sorted_array


def quicksort(xs, i=0, j=None, nc=None, history_distance=None):
    """
    Inspired by https://codereview.stackexchange.com/questions/272639/in-place-quicksort-algorithm-in-python

    Sort the array in place and return the number of comparisons.

    Parameters
    ----------
    xs: :class:`~numpy.ndarray`
        Array to sort.
    i: :class:`int`
        Index of the left boundary.
    j: :class:`int`
        Index of the right boundary.
    nc: :class:`list`
        A one-element list with the number of comparisons to update (for recursive calls).
    history_distance: :class:`list`
        History of the distance between the list and a sorted version (for recursive calls).

    Returns
    -------
    :class:`int`
        Number of comparisons performed.
    :class:`list`
        History of the distance between the list and a sorted version.

    Examples
    --------
        >>> my_xs = np.array([4, 1, 7, 6, 8, 2, 3, 5])
        >>> my_nc, my_history_distance = quicksort(my_xs)
        >>> my_nc
        16
        >>> my_history_distance
        [13, 13, 13, 13, 10, 9, 8, 8, 7, 6, 6, 6, 1, 1, 1, 0]
        >>> my_xs
        array([1, 2, 3, 4, 5, 6, 7, 8])

        >>> np.random.seed(42)
        >>> my_nc, my_history_distance = quicksort(np.random.permutation(100))
        >>> my_nc
        635
    """
    # Set optional arguments.
    if j is None:
        j = len(xs) - 1
    if nc is None:
        nc = [0]
    if history_distance is None:
        history_distance = []

    # Base case: do nothing if indexes have met or crossed.
    if not i < j:
        return nc[0]

    # Partition the sequence to enforce the quicksort invariant:
    # "small values" < pivot value <= "large values". The function
    # returns the index of the pivot value.
    pi = partition(xs, i, j, nc, history_distance)

    # Sort left side and right side.
    quicksort(xs, i=i, j=pi - 1, nc=nc, history_distance=history_distance)
    quicksort(xs, i=pi + 1, j=j, nc=nc, history_distance=history_distance)
    return nc[0], history_distance


def partition(xs, i, j, nc, history_distance):
    """

    Parameters
    ----------
    xs: :class:`~numpy.ndarray`
        Array to sort.
    i: :class:`int`
        Index of the left boundary.
    j: :class:`int`
        Index of the right boundary.
    nc: :class:`list`
        A one-element list with the number of comparisons (to update).
    history_distance: :class:`list`
        History of the distance between the list and a sorted version (to update).

    Returns
    -------
    :class:`int`
        Pivot index.
    """
    # Get the pivot value and initialize the partition boundary.
    pval = xs[i]
    pb = i + 1

    # Examine all values other than the pivot, swapping to enforce the
    # invariant. Every swap moves an observed "small" value to the left of the
    # boundary. "Large" values are left alone since they are already to the
    # right of the boundary.
    for k in range(i + 1, j + 1):
        nc[0] += 1
        if xs[k] < pval:
            swap(xs, k, pb)
            pb += 1
        history_distance.append(distance_to_sorted_array(xs))

    # Put pivot value between the two sides of the partition,
    # and return that location.
    swap(xs, i, pb - 1)
    history_distance[-1] = distance_to_sorted_array(xs)
    return pb - 1


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
