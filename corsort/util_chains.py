import numpy as np


def longest_chain_starting_at(leq, start_item):
    """
    Longest chain starting at a given item.

    Parameters
    ----------
    leq: :class:`~numpy.ndarray`.
        Matrix of size `(n_, n_)`. Coefficient (i, j) is
        +1 if we know that item i <= item j,
        -1 if we know that item i > item j,
        0 if we do not know the comparison between them.
    start_item: int
        Item which will be the smallest of the chain.

    Returns
    -------
    :class:`list`.
        The chain, from smallest to greatest element.

    Examples
    --------
        >>> my_leq = np.array([
        ...     [ 1,  1,  1,  1,  0,  0],
        ...     [-1,  1,  1,  0,  0,  0],
        ...     [-1, -1,  1,  0,  0,  0],
        ...     [-1,  0,  0,  1,  0,  0],
        ...     [ 0,  0,  0,  0,  1,  1],
        ...     [ 0,  0,  0,  0, -1,  1],
        ... ])
        >>> longest_chain_starting_at(my_leq, 0)
        [0, 1, 2]
    """
    greater_items = np.where(leq[:, start_item] == -1)[0]
    return [start_item] + max([
        longest_chain_starting_at(leq, second_item)
        for second_item in greater_items
    ], key=len, default=[])


def longest_chain(leq):
    """
    Longest chain.

    Parameters
    ----------
    leq: :class:`~numpy.ndarray`.
        Matrix of size `(n_, n_)`. Coefficient (i, j) is
        +1 if we know that item i <= item j,
        -1 if we know that item i > item j,
        0 if we do not know the comparison between them.

    Returns
    -------
    :class:`list`.
        The chain, from smallest to greatest element.

    Examples
    --------
        >>> my_leq = np.array([
        ...     [ 1,  1,  1,  1,  0,  0],
        ...     [-1,  1,  1,  0,  0,  0],
        ...     [-1, -1,  1,  0,  0,  0],
        ...     [-1,  0,  0,  1,  0,  0],
        ...     [ 0,  0,  0,  0,  1,  1],
        ...     [ 0,  0,  0,  0, -1,  1],
        ... ])
        >>> longest_chain(my_leq)
        [0, 1, 2]
    """
    minimal_elements = np.where(np.sum(leq == -1, axis=1) == 0)[0]
    return max([
        longest_chain_starting_at(leq, start_item)
        for start_item in minimal_elements
    ], key=len)


def greedy_chain_decomposition(leq):
    """
    Greedy chain decomposition.

    Find the longest chain. Then remove its elements from the graph, find the longest chain of the remaining vertices,
    and so on until exhaustion.

    Parameters
    ----------
    leq: :class:`~numpy.ndarray`.
        Matrix of size `(n_, n_)`. Coefficient (i, j) is
        +1 if we know that item i <= item j,
        -1 if we know that item i > item j,
        0 if we do not know the comparison between them.

    Returns
    -------
    :class:`list` of :class:`list`
        The chains, from the longest (found first during the algorithm) to the shortest (found last). Each
        chain is sorted from smallest item to greatest item.

    Examples
    --------
        >>> my_leq = np.array([
        ...     [ 1,  1,  1,  1,  0,  0],
        ...     [-1,  1,  1,  0,  0,  0],
        ...     [-1, -1,  1,  0,  0,  0],
        ...     [-1,  0,  0,  1,  0,  0],
        ...     [ 0,  0,  0,  0,  1,  1],
        ...     [ 0,  0,  0,  0, -1,  1],
        ... ])
        >>> greedy_chain_decomposition(my_leq)
        [[0, 1, 2], [4, 5], [3]]
    """
    leq_copy = leq.copy()
    n, _ = leq.shape
    number_of_recorded_items = 0
    chains = []
    while number_of_recorded_items < n:
        chain = longest_chain(leq_copy)
        chains.append(chain)
        number_of_recorded_items += len(chain)
        leq_copy[chain, :] = 0
        leq_copy[:, chain] = 0
        leq_copy[chain, chain] = -1
    return chains
