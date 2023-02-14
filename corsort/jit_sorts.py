from numba import njit
import numpy as np


@njit
def jit_corsort_borda(perm):
    """
    Corsort designed for low total number of comparison.
    Not efficient in terms of convergence trajectory.

    Parameters
    ----------
    perm: :class:`~numpy.ndarray`
        A random permutation.

    Returns
    -------
    states: :class:`list` of :class:`~numpy.ndarray`
        List of estimates of the sorted result.
    scores: :class:`list` of :class:`~numpy.ndarray`
        List of estimates of the importance of each item.
    comparisons: :class:`list` of :class:`tuple`
        List of performed comparison. Each element is a tuple (index of lower item, index of higher item).

    Examples
    --------

    >>> np.random.seed(42)
    >>> p = np.random.permutation(10)
    >>> st, sc, co = jit_corsort_borda(p)
    >>> st[0]
    array([8, 1, 5, 0, 7, 2, 9, 4, 3, 6])
    >>> st[10]
    array([0, 2, 1, 4, 5, 3, 6, 7, 8, 9])
    >>> st[-1]
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> p[:5]
    array([8, 1, 5, 0, 7])
    >>> sc[0][:5]
    array([0, 0, 0, 0, 0])
    >>> sc[10][:5]
    array([ 5, -3,  0, -5,  3])
    >>> sc[-1][:5]
    array([ 7, -7,  1, -9,  5])
    >>> len(co)
    22
    """
    n = len(perm)
    leq = np.eye(n, dtype=np.int8)
    pos = np.zeros(n, dtype=np.int_)
    info = np.zeros(n, dtype=np.int_)
    states = [perm[np.argsort(pos)]]
    scores = [pos.copy()]
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
        scores.append(pos.copy())
        states.append(perm[np.argsort(pos)])
    return states, scores, comparisons


@njit
def jit_corsort_drift_max_spaced(perm):
    """
    Corsort with drift core scorer, max-knowledge tie-break, and spaced outpus scorer.
    Currently, the best corsort for trajectory.

    Parameters
    ----------
    perm: :class:`~numpy.ndarray`
        A random permutation.

    Returns
    -------
    states: :class:`list` of :class:`~numpy.ndarray`
        List of estimates of the sorted result.
    scores: :class:`list` of :class:`~numpy.ndarray`
        List of estimates of the importance of each item.
    comparisons: :class:`list` of :class:`tuple`
        List of performed comparison. Each element is a tuple (index of lower item, index of higher item).

    Examples
    --------

    >>> np.random.seed(42)
    >>> p = np.random.permutation(10)
    >>> st, sc, co = jit_corsort_drift_max_spaced(p)
    >>> st[0]
    array([8, 1, 5, 0, 7, 2, 9, 4, 3, 6])
    >>> st[10]
    array([0, 1, 2, 4, 3, 6, 5, 7, 8, 9])
    >>> st[-1]
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> p[:5]
    array([8, 1, 5, 0, 7])
    >>> sc[0][:5]
    array([0.5, 0.5, 0.5, 0.5, 0.5])
    >>> sc[10][:5]
    array([0.83333333, 0.16666667, 0.66666667, 0.14285714, 0.75      ])
    >>> sc[-1][:5]
    array([0.81818182, 0.18181818, 0.54545455, 0.09090909, 0.72727273])
    >>> len(co)
    22
    """
    n = len(perm)
    leq = np.eye(n, dtype=np.int8)
    pos = np.zeros(n, dtype=np.int_)
    down = np.zeros(n, dtype=np.int_)+1
    info = np.zeros(n, dtype=np.int_)+2
    score = down/info
    scores = [score]
    states = [perm[np.argsort(score)]]
    comparisons = []
    i, j = 0, 1
    while True:
        diff = n
        xp = n+1
        for ii in range(n):
            for jj in range(ii + 1, n):
                if leq[ii, jj] == 0:
                    diff_ij = abs(pos[ii] - pos[jj])
                    if diff_ij > diff:
                        continue
                    xp_ij = max(info[ii], info[jj])
                    if diff_ij < diff or (diff_ij == diff and xp_ij < xp):
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
                        down[jj] += 1
                        pos[ii] -= 1
                        pos[jj] += 1
        score = down/info
        scores.append(score)
        states.append(perm[np.argsort(score)])
    return states, scores, comparisons


@njit
def jit_corsort_drift_plus_spaced(perm):
    """
    Corsort with drift core scorer, plus-knowledge tie-break, and spaced outpus scorer.

    Parameters
    ----------
    perm: :class:`~numpy.ndarray`
        A random permutation.

    Returns
    -------
    states: :class:`list` of :class:`~numpy.ndarray`
        List of estimates of the sorted result.
    scores: :class:`list` of :class:`~numpy.ndarray`
        List of estimates of the importance of each item.
    comparisons: :class:`list` of :class:`tuple`
        List of performed comparison. Each element is a tuple (index of lower item, index of higher item).

    Examples
    --------

    >>> np.random.seed(42)
    >>> p = np.random.permutation(10)
    >>> st, sc, co = jit_corsort_drift_plus_spaced(p)
    >>> st[0]
    array([8, 1, 5, 0, 7, 2, 9, 4, 3, 6])
    >>> st[10]
    array([0, 1, 2, 4, 3, 6, 5, 7, 8, 9])
    >>> st[-1]
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> p[:5]
    array([8, 1, 5, 0, 7])
    >>> sc[0][:5]
    array([0.5, 0.5, 0.5, 0.5, 0.5])
    >>> sc[10][:5]
    array([0.83333333, 0.16666667, 0.66666667, 0.14285714, 0.75      ])
    >>> sc[-1][:5]
    array([0.81818182, 0.18181818, 0.54545455, 0.09090909, 0.72727273])
    >>> len(co)
    21
    """
    n = len(perm)
    leq = np.eye(n, dtype=np.int8)
    pos = np.zeros(n, dtype=np.int_)
    down = np.zeros(n, dtype=np.int_)+1
    info = np.zeros(n, dtype=np.int_)+2
    score = down/info
    scores = [score]
    states = [perm[np.argsort(score)]]
    comparisons = []
    i, j = 0, 1
    while True:
        diff = n
        xp = 2*n
        for ii in range(n):
            for jj in range(ii + 1, n):
                if leq[ii, jj] == 0:
                    diff_ij = abs(pos[ii] - pos[jj])
                    if diff_ij > diff:
                        continue
                    xp_ij = info[ii] + info[jj]
                    if diff_ij < diff or (diff_ij == diff and xp_ij < xp):
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
                        down[jj] += 1
                        pos[ii] -= 1
                        pos[jj] += 1
        score = down/info
        scores.append(score)
        states.append(perm[np.argsort(score)])
    return states, scores, comparisons


@njit
def jit_corsort_drift_max_drift(perm):
    """
    Corsort with drift core scorer, max-knowledge tie-break, and drift outpus scorer.

    Parameters
    ----------
    perm: :class:`~numpy.ndarray`
        A random permutation.

    Returns
    -------
    states: :class:`list` of :class:`~numpy.ndarray`
        List of estimates of the sorted result.
    scores: :class:`list` of :class:`~numpy.ndarray`
        List of estimates of the importance of each item.
    comparisons: :class:`list` of :class:`tuple`
        List of performed comparison. Each element is a tuple (index of lower item, index of higher item).

    Examples
    --------

    >>> np.random.seed(42)
    >>> p = np.random.permutation(10)
    >>> st, sc, co = jit_corsort_drift_max_drift(p)
    >>> st[0]
    array([8, 1, 5, 0, 7, 2, 9, 4, 3, 6])
    >>> st[10]
    array([0, 1, 2, 4, 3, 6, 5, 7, 8, 9])
    >>> st[-1]
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> p[:5]
    array([8, 1, 5, 0, 7])
    >>> sc[0][:5]
    array([0, 0, 0, 0, 0])
    >>> sc[10][:5]
    array([ 4, -4,  2, -5,  2])
    >>> sc[-1][:5]
    array([ 7, -7,  1, -9,  5])
    >>> len(co)
    22
    """
    n = len(perm)
    leq = np.eye(n, dtype=np.int8)
    pos = np.zeros(n, dtype=np.int_)
    info = np.zeros(n, dtype=np.int_)
    score = pos.copy()
    scores = [score]
    states = [perm[np.argsort(score)]]
    comparisons = []
    i, j = 0, 1
    while True:
        diff = n
        xp = n+1
        for ii in range(n):
            for jj in range(ii + 1, n):
                if leq[ii, jj] == 0:
                    diff_ij = abs(pos[ii] - pos[jj])
                    if diff_ij > diff:
                        continue
                    xp_ij = max(info[ii], info[jj])
                    if diff_ij < diff or (diff_ij == diff and xp_ij < xp):
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
        score = pos.copy()
        scores.append(score)
        states.append(perm[np.argsort(score)])
    return states, scores, comparisons


@njit
def jit_corsort_drift_plus_drift(perm):
    """
    Corsort with drift core scorer, plus-knowledge tie-break, and drift outpus scorer.

    Parameters
    ----------
    perm: :class:`~numpy.ndarray`
        A random permutation.

    Returns
    -------
    states: :class:`list` of :class:`~numpy.ndarray`
        List of estimates of the sorted result.
    scores: :class:`list` of :class:`~numpy.ndarray`
        List of estimates of the importance of each item.
    comparisons: :class:`list` of :class:`tuple`
        List of performed comparison. Each element is a tuple (index of lower item, index of higher item).

    Examples
    --------

    >>> np.random.seed(42)
    >>> p = np.random.permutation(10)
    >>> st, sc, co = jit_corsort_drift_plus_drift(p)
    >>> st[0]
    array([8, 1, 5, 0, 7, 2, 9, 4, 3, 6])
    >>> st[10]
    array([0, 1, 2, 4, 3, 6, 5, 7, 8, 9])
    >>> st[-1]
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> p[:5]
    array([8, 1, 5, 0, 7])
    >>> sc[0][:5]
    array([0, 0, 0, 0, 0])
    >>> sc[10][:5]
    array([ 4, -4,  2, -5,  2])
    >>> sc[-1][:5]
    array([ 7, -7,  1, -9,  5])
    >>> len(co)
    21
    """
    n = len(perm)
    leq = np.eye(n, dtype=np.int8)
    pos = np.zeros(n, dtype=np.int_)
    info = np.zeros(n, dtype=np.int_)
    score = pos.copy()
    scores = [score]
    states = [perm[np.argsort(score)]]
    comparisons = []
    i, j = 0, 1
    while True:
        diff = n
        xp = 2*n
        for ii in range(n):
            for jj in range(ii + 1, n):
                if leq[ii, jj] == 0:
                    diff_ij = abs(pos[ii] - pos[jj])
                    if diff_ij > diff:
                        continue
                    xp_ij = info[ii] + info[jj]
                    if diff_ij < diff or (diff_ij == diff and xp_ij < xp):
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
        score = pos.copy()
        scores.append(score)
        states.append(perm[np.argsort(score)])
    return states, scores, comparisons


@njit
def jit_corsort_spaced_max_spaced(perm):
    """
    Corsort with spaced core scorer, max-knowledge tie-break, and spaced outpus scorer.

    Parameters
    ----------
    perm: :class:`~numpy.ndarray`
        A random permutation.

    Returns
    -------
    states: :class:`list` of :class:`~numpy.ndarray`
        List of estimates of the sorted result.
    scores: :class:`list` of :class:`~numpy.ndarray`
        List of estimates of the importance of each item.
    comparisons: :class:`list` of :class:`tuple`
        List of performed comparison. Each element is a tuple (index of lower item, index of higher item).

    Examples
    --------

    >>> np.random.seed(42)
    >>> p = np.random.permutation(10)
    >>> st, sc, co = jit_corsort_spaced_max_spaced(p)
    >>> st[0]
    array([8, 1, 5, 0, 7, 2, 9, 4, 3, 6])
    >>> st[10]
    array([0, 1, 2, 4, 3, 6, 5, 7, 8, 9])
    >>> st[-1]
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> p[:5]
    array([8, 1, 5, 0, 7])
    >>> sc[0][:5]
    array([0.5, 0.5, 0.5, 0.5, 0.5])
    >>> sc[10][:5]
    array([0.83333333, 0.16666667, 0.66666667, 0.14285714, 0.75      ])
    >>> sc[-1][:5]
    array([0.81818182, 0.18181818, 0.54545455, 0.09090909, 0.72727273])
    >>> len(co)
    23
    """
    n = len(perm)
    leq = np.eye(n, dtype=np.int8)
    down = np.zeros(n, dtype=np.int_)+1
    info = np.zeros(n, dtype=np.int_)+2
    score = down/info
    scores = [score]
    states = [perm[np.argsort(score)]]
    comparisons = []
    i, j = 0, 1
    while True:
        diff = n
        xp = n+1
        for ii in range(n):
            for jj in range(ii + 1, n):
                if leq[ii, jj] == 0:
                    diff_ij = abs(score[ii] - score[jj])
                    if diff_ij > diff:
                        continue
                    xp_ij = max(info[ii], info[jj])
                    if diff_ij < diff or (diff_ij == diff and xp_ij < xp):
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
                        down[jj] += 1
        score = down/info
        scores.append(score)
        states.append(perm[np.argsort(score)])
    return states, scores, comparisons


@njit
def jit_corsort_spaced_plus_spaced(perm):
    """
    Corsort with spaced core scorer, plus-knowledge tie-break, and spaced outpus scorer.

    Parameters
    ----------
    perm: :class:`~numpy.ndarray`
        A random permutation.

    Returns
    -------
    states: :class:`list` of :class:`~numpy.ndarray`
        List of estimates of the sorted result.
    scores: :class:`list` of :class:`~numpy.ndarray`
        List of estimates of the importance of each item.
    comparisons: :class:`list` of :class:`tuple`
        List of performed comparison. Each element is a tuple (index of lower item, index of higher item).

    Examples
    --------

    >>> np.random.seed(42)
    >>> p = np.random.permutation(10)
    >>> st, sc, co = jit_corsort_spaced_plus_spaced(p)
    >>> st[0]
    array([8, 1, 5, 0, 7, 2, 9, 4, 3, 6])
    >>> st[10]
    array([0, 1, 2, 4, 3, 6, 5, 7, 8, 9])
    >>> st[-1]
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> p[:5]
    array([8, 1, 5, 0, 7])
    >>> sc[0][:5]
    array([0.5, 0.5, 0.5, 0.5, 0.5])
    >>> sc[10][:5]
    array([0.83333333, 0.16666667, 0.66666667, 0.14285714, 0.75      ])
    >>> sc[-1][:5]
    array([0.81818182, 0.18181818, 0.54545455, 0.09090909, 0.72727273])
    >>> len(co)
    21
    """
    n = len(perm)
    leq = np.eye(n, dtype=np.int8)
    down = np.zeros(n, dtype=np.int_)+1
    info = np.zeros(n, dtype=np.int_)+2
    score = down/info
    scores = [score]
    states = [perm[np.argsort(score)]]
    comparisons = []
    i, j = 0, 1
    while True:
        diff = n
        xp = 2*n
        for ii in range(n):
            for jj in range(ii + 1, n):
                if leq[ii, jj] == 0:
                    diff_ij = abs(score[ii] - score[jj])
                    if diff_ij > diff:
                        continue
                    xp_ij = info[ii] + info[jj]
                    if diff_ij < diff or (diff_ij == diff and xp_ij < xp):
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
                        down[jj] += 1
        score = down/info
        scores.append(score)
        states.append(perm[np.argsort(score)])
    return states, scores, comparisons


@njit
def jit_corsort_spaced_max_drift(perm):
    """
    Corsort with spaced core scorer, max-knowledge tie-break, and drift outpus scorer.

    Parameters
    ----------
    perm: :class:`~numpy.ndarray`
        A random permutation.

    Returns
    -------
    states: :class:`list` of :class:`~numpy.ndarray`
        List of estimates of the sorted result.
    scores: :class:`list` of :class:`~numpy.ndarray`
        List of estimates of the importance of each item.
    comparisons: :class:`list` of :class:`tuple`
        List of performed comparison. Each element is a tuple (index of lower item, index of higher item).

    Examples
    --------

    >>> np.random.seed(42)
    >>> p = np.random.permutation(10)
    >>> st, sc, co = jit_corsort_spaced_max_drift(p)
    >>> st[0]
    array([8, 1, 5, 0, 7, 2, 9, 4, 3, 6])
    >>> st[10]
    array([0, 1, 2, 4, 3, 6, 5, 7, 8, 9])
    >>> st[-1]
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> p[:5]
    array([8, 1, 5, 0, 7])
    >>> sc[0][:5]
    array([0, 0, 0, 0, 0])
    >>> sc[10][:5]
    array([ 4, -4,  2, -5,  2])
    >>> sc[-1][:5]
    array([ 7, -7,  1, -9,  5])
    >>> len(co)
    23
    """
    n = len(perm)
    leq = np.eye(n, dtype=np.int8)
    pos = np.zeros(n, dtype=np.int_)
    down = np.zeros(n, dtype=np.int_)+1
    info = np.zeros(n, dtype=np.int_)+2
    est = down/info
    score = pos.copy()
    scores = [score]
    states = [perm[np.argsort(score)]]
    comparisons = []
    i, j = 0, 1
    while True:
        diff = n
        xp = n+1
        for ii in range(n):
            for jj in range(ii + 1, n):
                if leq[ii, jj] == 0:
                    diff_ij = abs(est[ii] - est[jj])
                    if diff_ij > diff:
                        continue
                    xp_ij = max(info[ii], info[jj])
                    if diff_ij < diff or (diff_ij == diff and xp_ij < xp):
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
                        down[jj] += 1
                        pos[ii] -= 1
                        pos[jj] += 1
        est = down/info
        score = pos.copy()
        scores.append(score)
        states.append(perm[np.argsort(score)])
    return states, scores, comparisons


@njit
def jit_corsort_spaced_plus_drift(perm):
    """
    Corsort with spaced core scorer, plus-knowledge tie-break, and drift outpus scorer.

    Parameters
    ----------
    perm: :class:`~numpy.ndarray`
        A random permutation.

    Returns
    -------
    states: :class:`list` of :class:`~numpy.ndarray`
        List of estimates of the sorted result.
    scores: :class:`list` of :class:`~numpy.ndarray`
        List of estimates of the importance of each item.
    comparisons: :class:`list` of :class:`tuple`
        List of performed comparison. Each element is a tuple (index of lower item, index of higher item).

    Examples
    --------

    >>> np.random.seed(42)
    >>> p = np.random.permutation(10)
    >>> st, sc, co = jit_corsort_spaced_plus_drift(p)
    >>> st[0]
    array([8, 1, 5, 0, 7, 2, 9, 4, 3, 6])
    >>> st[10]
    array([0, 1, 2, 4, 3, 6, 5, 7, 8, 9])
    >>> st[-1]
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> p[:5]
    array([8, 1, 5, 0, 7])
    >>> sc[0][:5]
    array([0, 0, 0, 0, 0])
    >>> sc[10][:5]
    array([ 4, -4,  2, -5,  2])
    >>> sc[-1][:5]
    array([ 7, -7,  1, -9,  5])
    >>> len(co)
    21
    """
    n = len(perm)
    leq = np.eye(n, dtype=np.int8)
    pos = np.zeros(n, dtype=np.int_)
    down = np.zeros(n, dtype=np.int_)+1
    info = np.zeros(n, dtype=np.int_)+2
    est = down/info
    score = pos.copy()
    scores = [score]
    states = [perm[np.argsort(score)]]
    comparisons = []
    i, j = 0, 1
    while True:
        diff = n
        xp = 2*n
        for ii in range(n):
            for jj in range(ii + 1, n):
                if leq[ii, jj] == 0:
                    diff_ij = abs(est[ii] - est[jj])
                    if diff_ij > diff:
                        continue
                    xp_ij = info[ii] + info[jj]
                    if diff_ij < diff or (diff_ij == diff and xp_ij < xp):
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
                        down[jj] += 1
                        pos[ii] -= 1
                        pos[jj] += 1
        est = down/info
        score = pos.copy()
        scores.append(score)
        states.append(perm[np.argsort(score)])
    return states, scores, comparisons


@njit
def heapify(arr, n, i, states, scores, comparisons):
    """
    Based on a code by Mohit Kumra:
    https://www.geeksforgeeks.org/python-program-for-heap-sort/
    """
    largest = i  # Initialize largest as root
    left = 2 * i + 1  # left = 2*i + 1
    right = 2 * i + 2  # right = 2*i + 2

    # See if left child of root exists and is
    # greater than root

    if left < n:
        if arr[i] < arr[left]:
            largest = left
            comparisons.append((i, left))
        else:
            comparisons.append((left, i))
        scores.append(0)
        states.append(arr.copy())

    # See if right child of root exists and is
    # greater than root

    if right < n:
        if arr[largest] < arr[right]:
            largest = right
            comparisons.append((largest, right))
        else:
            comparisons.append((right, largest))
        scores.append(0)
        states.append(arr.copy())

    # Change root, if needed
    if largest != i:
        (arr[i], arr[largest]) = (arr[largest], arr[i])  # swap

        # Heapify the root.
        heapify(arr, n, largest, states, scores, comparisons)


# The main function to sort an array of given size
@njit
def jit_heapsort(arr):
    """
    Heap sort.

    Parameters
    ----------
    arr:class:`~numpy.ndarray`
        A random permutation.

    Returns
    -------
    states: :class:`list` of :class:`~numpy.ndarray`
        List of estimates of the sorted result.
    scores: :class:`list` of :class:`~numpy.ndarray`
        List of estimates of the importance of each item.
    comparisons: :class:`list` of :class:`tuple`
        List of performed comparison. Each element is a tuple (index of lower item, index of higher item).

    Examples
    --------
    >>> np.random.seed(42)
    >>> p = np.random.permutation(10)
    >>> st, sc, co = jit_heapsort(p)
    """
    n = len(arr)
    states = [arr.copy()]
    scores = [0]
    comparisons = [(0, 0) for _ in range(0)]

    # Build a max-heap.
    # Since last parent will be at ((n//2)-1) we can start at that location.

    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i, states, scores, comparisons)

    # One by one extract elements

    for i in range(n - 1, 0, -1):
        (arr[i], arr[0]) = (arr[0], arr[i])  # swap
        heapify(arr, i, 0, states, scores, comparisons)
    return states, scores, comparisons
