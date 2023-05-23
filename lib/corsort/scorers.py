import numpy as np


def scorer_delta(leq):
    """
    Scorer delta.

    Parameters
    ----------
    leq: :class:`~numpy.ndarray`.
        Matrix of size `(n_, n_)`. Coefficient (i, j) is
        +1 if we know that item i <= item j,
        -1 if we know that item i > item j,
        0 if we do not know the comparison between them.

    Returns
    -------
    :class:`~numpy.ndarray`
        Score for each item.

    Examples
    --------
    Up to 1, this is just the sum of the column of the leq matrix:

        >>> my_leq = np.array([
        ...     [ 1,  1,  1,  1],
        ...     [-1,  1, -1, -1],
        ...     [-1,  1,  1,  0],
        ...     [-1,  1,  0,  1],
        ... ])
        >>> scorer_delta(my_leq)
        array([-3,  3,  0,  0])

    We can deduce the Borda score from it:

        >>> n = my_leq.shape[0]
        >>> (scorer_delta(my_leq) + n - 1) / 2
        array([0. , 3. , 1.5, 1.5])
    """
    return np.sum(leq, axis=0) - 1


def scorer_rho(leq):
    """
    Scorer rho.

    Parameters
    ----------
    leq: :class:`~numpy.ndarray`.
        Matrix of size `(n_, n_)`. Coefficient (i, j) is
        +1 if we know that item i <= item j,
        -1 if we know that item i > item j,
        0 if we do not know the comparison between them.

    Returns
    -------
    :class:`~numpy.ndarray`
        Score for each item.

    Examples
    --------
        >>> my_leq = np.array([
        ...     [ 1,  1,  1,  1],
        ...     [-1,  1, -1, -1],
        ...     [-1,  1,  1,  0],
        ...     [-1,  1,  0,  1],
        ... ])
        >>> scorer_rho(my_leq)
        array([0.25, 4.  , 1.  , 1.  ])
    """
    n_ancestors = np.sum(leq == 1, axis=1)
    n_descendants = np.sum(leq == 1, axis=0)
    return n_descendants / n_ancestors
