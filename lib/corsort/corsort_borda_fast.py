from numba import njit
import numpy as np
from corsort.entropy_bound import entropy_bound
from corsort.distance_to_sorted_array import distance_to_sorted_array


@njit
def jit_corsort(perm):
    n = len(perm)
    leq = np.eye(n, dtype=np.bool_)
    pos = np.zeros(n, dtype=np.int_)
    info = np.zeros(n, dtype=np.int_)
    distances = [perm[np.argsort(pos)]]
    while True:
        diff = n
        xp = 0
        for ii in range(n):
            for jj in range(ii+1, n):
                if (not leq[ii, jj]) and (not leq[jj, ii]):
                    diff_ij = abs(pos[ii]-pos[jj])
                    xp_ij = info[ii] + info[jj]
                    if (diff_ij < diff) or (diff_ij == diff and xp_ij > xp):
                        diff = diff_ij
                        xp = xp_ij
                        i, j = ii, jj
        if diff == n:
            break
        # TODO: Check whether there is an edge case to deal with, where i/j might not be assigned before getting here.
        # noinspection PyUnboundLocalVariable
        if perm[i] > perm[j]:
            i, j = j, i
        for ii in range(n):
            for jj in range(n):
                if leq[ii, i] and leq[j, jj] and (not leq[ii, jj]):
                    leq[ii, jj] = True
                    info[ii] += 1
                    info[jj] += 1
                    pos[ii] -= 1
                    pos[jj] += 1
        distances.append(perm[np.argsort(pos)])
    return distances


def corsort_borda_fast(perm, compute_history=False):
    """
    A fast, direct, implementation of the Borda Corsort.

    Parameters
    ----------
    perm: :class:`numpy.ndarray`
        Input (random) permutation
    compute_history: :class:`bool`
        If True, then compute the history of the distance to the sorted array.

    Returns
    -------
    d: :class:`int`
        Number of permutations required to sort the input
    distances: :class:`list`
        Evolution of distances to target.

    Examples
    --------

    >>> np.random.seed(22)
    >>> n = 15
    >>> p = np.random.permutation(n)
    >>> d, dists =corsort_borda_fast(p, compute_history=True)
    >>> d
    43
    >>> entropy_bound(n) # doctest: +ELLIPSIS
    40.869...
    >>> dists # doctest: +NORMALIZE_WHITESPACE
    [55, 42, 51, 49, 49, 48, 40, 39, 33, 29, 29, 29, 28, 28, 28, 26, 26, 21, 20, 16, 14, 11, 10, 9, 8, 10,
    8, 7, 6, 7, 5, 4, 4, 5, 4, 3, 4, 3, 3, 2, 2, 0, 1, 0]
    """
    states = jit_corsort(perm)
    history = [distance_to_sorted_array(state) for state in states] if compute_history else []
    return len(states)-1, history
    # n = len(perm)
    # an = np.eye(n, dtype=bool)
    # pos = np.sum(an, axis=0) - np.sum(an, axis=1)
    # distances = [distance_to_sorted_array(perm[np.argsort(pos)])]
    # while True:
    #     pos_matrix = 2*n-np.abs(pos[np.newaxis, :] - pos[:, np.newaxis])
    #     pos_matrix[an] = 0
    #     pos_matrix[an.T] = 0
    #     i, j = np.unravel_index(pos_matrix.argmax(), pos_matrix.shape)
    #     if pos_matrix[i, j] == 0:
    #         break
    #     if perm[i] < perm[j]:
    #         an[np.ix_(an[:, i], an[j, :])] = True
    #     else:
    #         an[np.ix_(an[:, j], an[i, :])] = True
    #     pos = np.sum(an, axis=0) - np.sum(an, axis=1)
    #     distances.append(distance_to_sorted_array(perm[np.argsort(pos)]))
    # return len(distances)-1, distances
