def merge_sort_dfs(collection, lt):
    """

    Parameters
    ----------
    collection: class: `list`
                        A list to sort
    lt: class: `function`
                a function that takes two elements x and y return the boolean x < y

    Returns
    -------
    None

    Examples:
    ---------

    >>> L=[7, 3, 2, 1, 4, 6, 0, 5]
    >>> merge_sort_dfs(L, None)
    >>> L
    [0, 1, 2, 3, 4, 5, 6, 7]

    """
    if lt is None:
        def lt(x, y):
            return x < y
    if len(collection) > 1:
        middle = len(collection)//2
        left_coll, right_coll = collection[:middle], collection[middle:]
        merge_sort_dfs(left_coll, lt)
        merge_sort_dfs(right_coll, lt)
        i, j, k = 0, 0, 0
        while i < len(left_coll) and j < len(right_coll):
            if lt(left_coll[i], right_coll[j]):
                collection[k] = left_coll[i]
                i += 1
            else:
                collection[k] = right_coll[j]
                j += 1
            k += 1
        while i < len(left_coll):
            collection[k] = left_coll[i]
            i += 1
            k += 1
        while j < len(right_coll):
            collection[k] = right_coll[j]
            j += 1
            k += 1
