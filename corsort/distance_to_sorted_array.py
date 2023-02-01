import numpy as np
import scipy


def _distance_to_sorted_array_old(xs):
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
        >>> _distance_to_sorted_array_old(np.array([2, 1, 3]))
        1
        >>> _distance_to_sorted_array_old(np.array([4, 1, 7, 6, 0, 8, 2, 3, 5]))
        17
    """
    return np.sum(np.tril(xs[:, np.newaxis] < xs[np.newaxis, :]))


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
    n = len(xs)
    res = scipy.stats.kendalltau(xs, np.arange(n))
    return round(n*(n-1)*(1-res.statistic)/4)
