import numpy as np


def entropy_bound(n):
    """
    Gives an approximation of the information theoretical lower bound of the number of comparisons
    required to sort n items.

    Cf. https://en.wikipedia.org/wiki/Comparison_sort

    Parameters
    ----------
    n: :class:`int`
        Number of items to sort.

    Returns
    -------
    :class:`float`
        A lower bound.

    Examples
    --------
    >>> print(f"{entropy_bound(10):.1f}")
    21.8
    >>> print(f"{entropy_bound(100):.1f}")
    524.8
    >>> print(f"{entropy_bound(1000):.1f}")
    8529.4
    """
    return n * (np.log2(n) - 1 / np.log(2)) + .5 * np.log2(2 * np.pi * n)
