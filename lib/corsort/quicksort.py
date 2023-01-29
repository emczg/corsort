import numpy as np
from corsort.distance_to_sorted_array import distance_to_sorted_array


def quicksort(xs, i=0, j=None, nc=None, history_distance=None, compute_history=False):
    """
    Inspired by https://codereview.stackexchange.com/questions/272639/in-place-quicksort-algorithm-in-python

    Sort the array in place. Return the number of comparisons, and the history of the distance to the sorted list.

    Parameters
    ----------
    xs: :class:`~numpy.ndarray`
        Array to sort.
    i: :class:`int`
        Index of the left boundary.
    j: :class:`int`
        Index of the right boundary.
    nc: :class:`list`
        A one-element list with the number of comparisons (to update in recursive calls).
    history_distance: :class:`list`
        History of the distance between the list and the sorted list (to update in recursive calls).

    Returns
    -------
    :class:`int`
        Number of comparisons performed.
    :class:`list`
        History of the distance between the list and the sorted list.

    Examples
    --------
        >>> my_xs = np.array([4, 1, 7, 6, 0, 8, 2, 3, 5])
        >>> my_nc, my_history_distance = quicksort(my_xs, compute_history=True)
        >>> my_xs
        array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        >>> my_nc
        17
        >>> my_history_distance
        [16, 16, 16, 12, 12, 10, 6, 6, 5, 5, 5, 5, 4, 3, 2, 2, 0]

        >>> np.random.seed(42)
        >>> my_nc, my_history_distance = quicksort(np.random.permutation(100))
        >>> my_nc
        659
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
    pivot_index = partition(xs, i, j, nc, history_distance, compute_history)

    # Sort left side and right side.
    quicksort(xs, i=i, j=pivot_index - 1, nc=nc, history_distance=history_distance,
              compute_history=compute_history)
    quicksort(xs, i=pivot_index + 1, j=j, nc=nc, history_distance=history_distance,
              compute_history=compute_history)
    return nc[0], history_distance


def partition(xs, i, j, nc, history_distance, compute_history=False):
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
        History of the distance between the list and the sorted list (to update).

    Returns
    -------
    :class:`int`
        Pivot index.

    Examples
    --------
        >>> my_xs = np.array([4, 1, 7, 6, 0, 8, 2, 3, 5])
        >>> partition(my_xs, i=0, j=len(my_xs) - 1, nc=[0], history_distance=[])
        4
        >>> my_xs
        array([1, 0, 2, 3, 4, 8, 6, 7, 5])
    """
    pivot_value = xs[i]
    pivot_index = i
    for k in range(i + 1, j + 1):
        nc[0] += 1
        if xs[k] < pivot_value:
            if k > pivot_index + 1:
                xs[pivot_index], xs[pivot_index + 1], xs[k] = xs[k], pivot_value, xs[pivot_index + 1]
            else:
                xs[pivot_index], xs[k] = xs[k], pivot_value
            pivot_index += 1
        if compute_history:
            history_distance.append(distance_to_sorted_array(xs))
    return pivot_index
