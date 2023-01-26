import numpy as np


def distance_to_sorted_array(xs):
    """
    Kendall-tau distance to the sorted array.

    Parameters
    ----------
    xs: :class:`~numpy.ndarray`
        The array.

    Returns
    -------
    :class:`int`
        The distance

    Examples
    --------
        >>> distance_to_sorted_array(np.array([2, 1, 3]))
        1
        >>> distance_to_sorted_array(np.array([4, 1, 7, 6, 0, 8, 2, 3, 5]))
        17
    """
    return np.sum(np.tril(xs[:, np.newaxis] < xs[np.newaxis, :]))
