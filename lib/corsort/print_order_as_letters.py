import string


def print_order_as_letters(order):
    """
    Print a small list of integers as letters.

    Parameters
    ----------
    order: :class:`list`
        Size should be 26 at most.

    Returns
    -------
    str

    Examples
    --------
        >>> print_order_as_letters([4, 2, 3, 1, 0])
        (ecdba)
    """
    print("(" + "".join([string.ascii_lowercase[i] for i in order]) + ")")
