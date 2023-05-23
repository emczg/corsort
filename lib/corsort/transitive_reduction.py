import numpy as np
from itertools import permutations


def transitive_reduction(leq):
    """
    Transitive reduction of a `leq` matrix.

    Parameters
    ----------
    leq: :class:`~numpy.ndarray`.
        Matrix of size `(n_, n_)`. Coefficient (i, j) is
        +1 if we know that item i <= item j,
        -1 if we know that item i > item j,
        0 if we do not know the comparison between them.

    Returns
    -------
    :class:`list`
        List of edges of the transitive reduction.

    Examples
    --------
        >>> my_leq = np.array([
        ...     [1, 1, 1],
        ...     [0, 1, 1],
        ...     [0, 0, 1],
        ... ])
        >>> transitive_reduction(my_leq)
        [(0, 1), (1, 2)]
    """
    mask_keep = (leq == 1)
    comparisons = [(i, j) for i, j in zip(*np.where(leq == 1)) if i != j]
    for (i, j), (k, l) in permutations(comparisons, 2):
        if j == k:
            mask_keep[i, l] = False
    comparisons = [(i, j) for i, j in zip(*np.where(mask_keep)) if i != j]
    return comparisons
