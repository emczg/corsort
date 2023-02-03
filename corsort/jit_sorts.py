from numba import njit
import numpy as np


@njit
def jit_corsort_borda(perm):
    n = len(perm)
    leq = np.eye(n, dtype=np.int8)
    pos = np.zeros(n, dtype=np.int_)
    info = np.zeros(n, dtype=np.int_)
    states = [perm[np.argsort(pos)]]
    scores = [pos[:]]
    comparisons = []
    i, j = 0, 1
    while True:
        diff = n
        xp = 0
        for ii in range(n):
            for jj in range(ii + 1, n):
                if leq[ii, jj] == 0:
                    diff_ij = abs(pos[ii] - pos[jj])
                    if diff_ij > diff:
                        continue
                    xp_ij = info[ii] + info[jj]
                    if diff_ij < diff or (diff_ij == diff and xp_ij > xp):
                        diff = diff_ij
                        xp = xp_ij
                        i, j = ii, jj
        if diff == n:
            break
        if perm[i] > perm[j]:
            i, j = j, i
        comparisons.append((i, j))
        for ii in range(n):
            if leq[ii, i] > 0:
                for jj in range(n):
                    if leq[j, jj] > 0 and leq[ii, jj] == 0:
                        leq[ii, jj] = 1
                        leq[jj, ii] = -1
                        info[ii] += 1
                        info[jj] += 1
                        pos[ii] -= 1
                        pos[jj] += 1
        scores.append(pos[:])
        states.append(perm[np.argsort(pos)])
    return states, scores, comparisons
