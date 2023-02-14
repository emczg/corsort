from numba import njit
import numpy as np


@njit
def scorer_spaced(n, downs, ups):
    """
    Estimates scores of nodes by dividing the number of the descendants by the size of the family plus one.
    The rational is to consider that the family should be in average evenly spaced in the sorted result.

    Parameters
    ----------
    n: :class:`int`
        Number of items to sort
    downs: :class:`~numpy.ndarray`
        Indices of the low value of a sequence of performed comparisons
    ups: :class:`~numpy.ndarray`
        Indices of the high value of a sequence of performed comparisons

    Returns
    -------
    :class:`~numpy.ndarray`
        An array that represents, for each new performed comparison,
        some item scores that estimate the final position of items.

    Examples
    --------

    >>> my_n = 5
    >>> my_downs = np.array([0, 1, 2])
    >>> my_ups = np.array([2, 2, 3])
    >>> scorer_spaced(my_n, my_downs, my_ups)
    array([[0.5       , 0.5       , 0.5       , 0.5       , 0.5       ],
           [0.33333333, 0.5       , 0.66666667, 0.5       , 0.5       ],
           [0.33333333, 0.33333333, 0.75      , 0.5       , 0.5       ],
           [0.25      , 0.25      , 0.6       , 0.8       , 0.5       ]])
    """
    ncp = len(ups) + 1
    leq = np.eye(n, dtype=np.int8)
    down = np.zeros(n, dtype=np.int_) + 1
    tot = np.zeros(n, dtype=np.int_) + 2
    res = np.zeros((ncp, n))
    res[0, :] = down / tot
    for k in range(len(ups)):
        i, j = downs[k], ups[k]
        for ii in range(n):
            if leq[ii, i] > 0:
                for jj in range(n):
                    if leq[j, jj] > 0 and leq[ii, jj] == 0:
                        leq[ii, jj] = 1
                        leq[jj, ii] = -1
                        tot[ii] += 1
                        tot[jj] += 1
                        down[jj] += 1
        res[(k + 1), :] = down / tot
    return res


@njit
def scorer_drift(n, downs, ups):
    """
    Estimates scores of nodes by the difference between the numbers of descendants and ascendants.
    The rational is to consider that an item should be in average halfway between
    its highest and lowest possible values.

    Parameters
    ----------
    n: :class:`int`
        Number of items to sort
    downs: :class:`~numpy.ndarray`
        Indices of the low value of a sequence of performed comparisons
    ups: :class:`~numpy.ndarray`
        Indices of the high value of a sequence of performed comparisons

    Returns
    -------
    :class:`~numpy.ndarray`
        An array that represents, for each new performed comparison,
        some item scores that estimate the final position of items.

    Examples
    --------

    >>> my_n = 5
    >>> my_downs = np.array([0, 1, 2])
    >>> my_ups = np.array([2, 2, 3])
    >>> scorer_drift(my_n, my_downs, my_ups)
    array([[ 0,  0,  0,  0,  0],
           [-1,  0,  1,  0,  0],
           [-1, -1,  2,  0,  0],
           [-2, -2,  1,  3,  0]])
    """
    ncp = len(ups) + 1
    leq = np.eye(n, dtype=np.int8)
    pos = np.zeros(n, dtype=np.int_)
    res = np.zeros((ncp, n), dtype=np.int_)
    for k in range(len(ups)):
        i, j = downs[k], ups[k]
        for ii in range(n):
            if leq[ii, i] > 0:
                for jj in range(n):
                    if leq[j, jj] > 0 and leq[ii, jj] == 0:
                        leq[ii, jj] = 1
                        leq[jj, ii] = -1
                        pos[ii] -= 1
                        pos[jj] += 1
        res[(k + 1), :] = pos
    return res
