import numpy as np


def _sub_step(split_pointer_list):
    """
    Compute the indices of the boundaries for the next sub-step of bottom-up (BFS) merge sort.

    Parameters
    ----------
    split_pointer_list: :class:`~numpy.ndarray`
        List of split pointers for the current step.

    Returns
    -------
    :class:`~numpy.ndarray`
        List of split pointers for the sub-step.

    Examples
    --------
        >>> my_indices = np.array([0, 9])
        >>> my_indices = _sub_step(my_indices)
        >>> my_indices
        array([0, 4, 9])
        >>> my_indices = _sub_step(my_indices)
        >>> my_indices
        array([0, 2, 4, 6, 9])
        >>> my_indices = _sub_step(my_indices)
        >>> my_indices
        array([0, 1, 2, 3, 4, 5, 6, 7, 9])
        >>> my_indices = _sub_step(my_indices)
        >>> my_indices
        array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 9])
    """
    list_indices_sub_step = []
    for i, j in zip(split_pointer_list[:-1], split_pointer_list[1:]):
        list_indices_sub_step.append(i)
        list_indices_sub_step.append((i + j) // 2)
    list_indices_sub_step.append(split_pointer_list[-1])
    return np.array(list_indices_sub_step)


def split_pointer_lists(n):
    """
    Compute the indices of the boundaries for all the steps of bottom-up (BFS) merge sort.

    Parameters
    ----------
    n: :class:`integer`
        Size of the list.

    Returns
    -------
    :class:`list` of :class:`~numpy.ndarray`
        For each step, list of indices for the step.

    Examples
    --------
        >>> split_pointer_lists(n=9)  # doctest: +NORMALIZE_WHITESPACE
        [array([0, 4, 9]),
        array([0, 2, 4, 6, 9]),
        array([0, 1, 2, 3, 4, 5, 6, 7, 9]),
        array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 8, 9])]
    """
    split_pointer_list = np.array([0, n])
    result = []
    while np.max(split_pointer_list[1:] - split_pointer_list[:-1]) > 1:
        split_pointer_list = _sub_step(split_pointer_list)
        result.append(split_pointer_list)
    return result
