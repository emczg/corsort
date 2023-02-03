from numba import njit
import numpy as np


@njit
def scorer_spaced(n, downs, ups):
    ncp = len(ups) + 1
    leq = np.eye(n, dtype=np.int8)
    down = np.zeros(n, dtype=np.int_) + 1
    tot = np.zeros(n, dtype=np.int_) + 2
    res = np.zeros((ncp, n))
    res[0, :] = down/tot
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
        res[(k+1), :] = down/tot
    return res
