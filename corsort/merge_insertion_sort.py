def merge_insertion_sort(collection):
    """Merge-insertion sort (Ford-Johnson).

    Inspired by https://www.w3resource.com/python-exercises/data-structures-and-algorithms/python-search-and-sorting-exercise-39.php

    Parameters
    ----------
    collection: :class:`list`
        Some mutable ordered collection with comparable items inside.

    Returns
    -------
    :class:`list`
        The same collection ordered by ascending order.

    Examples
    --------
        >>> merge_insertion_sort([4, 1, 7, 6, 0, 8, 2, 3, 5])
        [0, 1, 2, 3, 4, 5, 6, 7, 8]
        >>> merge_insertion_sort([0, 5, 3, 2, 2])
        [0, 2, 2, 3, 5]
        >>> merge_insertion_sort([99])
        [99]
        >>> merge_insertion_sort([-2, -5, -45])
        [-45, -5, -2]
    """

    if len(collection) <= 1:
        return collection

    """
    Group the items into two pairs, and leave one element if there is a last odd item.
    Example: [999, 100, 75, 40, 10000]
    After this step:
      two_paired_list : [[100, 999], [40, 75]]
      has_last_odd_item : True
    """
    two_paired_list = []
    has_last_odd_item = False
    for i in range(0, len(collection), 2):
        if i == len(collection) - 1:
            has_last_odd_item = True
        else:
            if collection[i] < collection[i + 1]:
                two_paired_list.append([collection[i], collection[i + 1]])
            else:
                two_paired_list.append([collection[i + 1], collection[i]])

    """
    Sort two_paired_list according to the first element of each pair.
    Example: [[100, 999], [40, 75]]
    After this step:
      sorted_list_2d = [[40, 75], [100, 999]]
    """
    sorted_list_2d = sortlist_2d(two_paired_list)

    """
    40 < 100 is sure because it has already been sorted.
    Generate the sorted_list of them so that you can avoid unnecessary comparison.
    Example:
           group0 group1
           40     100
           75     999
        ->
           group0 group1
           [40,   100]
           75     999
    """
    result = [i[0] for i in sorted_list_2d]

    """
    100 < 999 is sure because it has already been sorted.
    Put 999 in last of the sorted_list so that you can avoid unnecessary comparison.
    Example:
           group0 group1
           [40,   100]
           75     999
        ->
           group0 group1
           [40,   100,   999]
           75
    """
    result.append(sorted_list_2d[-1][1])

    """
    Insert the last odd item left if there is.
    Example:
           group0 group1
           [40,   100,   999]
           75
        ->
           group0 group1
           [40,   100,   999,   10000]
           75
    """
    if has_last_odd_item:
        pivot = collection[-1]
        result = binary_search_insertion(result, pivot)

    """
    Insert the remaining items.
    In this case, 40 < 75 is sure because it has already been sorted.
    Therefore, you only need to insert 75 into [100, 999, 10000],
    so that you can avoid unnecessary comparison.
    Example:
           group0 group1
           [40,   100,   999,   10000]
            ^ You don't need to compare with this as 40 < 75 is already sure.
           75
        ->
           [40,   75,    100,   999,   10000]
    """
    is_last_odd_item_inserted_at_this_index_or_before = False
    for i in range(len(sorted_list_2d) - 1):
        if has_last_odd_item and result[i] == collection[-1]:
            is_last_odd_item_inserted_at_this_index_or_before = True
        pivot = sorted_list_2d[i][1]
        # If last_odd_item is inserted before the item's index,
        # you should forward index one more.
        if is_last_odd_item_inserted_at_this_index_or_before:
            result = result[: i + 2] + binary_search_insertion(result[i + 2:], pivot)
        else:
            result = result[: i + 1] + binary_search_insertion(result[i + 1:], pivot)

    return result


def binary_search_insertion(sorted_list, item):
    """
    Insert an_ item in a sorted list by binary search.

    Parameters
    ----------
    sorted_list: :class:`list`
        A sorted list.
    item: object
        The item to insert.

    Returns
    -------
    :class:`list`
        The sorted list with the inserted item.

    Examples
    --------
        >>> binary_search_insertion([1, 12, 45, 51, 69, 99], 42)
        [1, 12, 42, 45, 51, 69, 99]
    """
    left = 0
    right = len(sorted_list) - 1
    while left <= right:
        middle = (left + right) // 2
        if left == right:
            if sorted_list[middle] < item:
                left = middle + 1
            break
        elif sorted_list[middle] < item:
            left = middle + 1
        else:
            right = middle - 1
    sorted_list.insert(left, item)
    return sorted_list


def sortlist_2d(list_2d):
    """Sort a list of pairs according to their first elements.

    Parameters
    ----------
    list_2d: :class:`list`
        List of pairs.

    Returns
    -------
    :class:`list`
        The list, sorted according to the first element of each pair.

    Examples
    --------
        >>> my_lst = [[1, 43], [3, 35], [4, 11], [0, 12], [2, 28], [5, 18]]
        >>> sortlist_2d(my_lst)
        [[0, 12], [1, 43], [2, 28], [3, 35], [4, 11], [5, 18]]
    """
    length = len(list_2d)
    if length <= 1:
        return list_2d
    middle = length // 2
    return merge(sortlist_2d(list_2d[:middle]), sortlist_2d(list_2d[middle:]))


def merge(left, right):
    """
    Merge two lists of pairs that are already sorted according to their first element.

    Parameters
    ----------
    left: :class:`list`
        List of pairs.
    right: :class:`list`
        List of pairs.

    Returns
    -------
    :class:`list`
        The merged list, sorted according to the first element of each pair.

    Examples
    --------
        >>> my_left = [[1, 43], [3, 35], [4, 11]]
        >>> my_right = [[0, 12], [2, 28], [5, 18]]
        >>> merge(my_left, my_right)
        [[0, 12], [1, 43], [2, 28], [3, 35], [4, 11], [5, 18]]
    """
    result = []
    while left and right:
        if left[0][0] < right[0][0]:
            result.append(left.pop(0))
        else:
            result.append(right.pop(0))
    return result + left + right
