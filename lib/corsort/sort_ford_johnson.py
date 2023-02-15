import numpy as np
from corsort.sort import Sort


class SortFordJohnson(Sort):
    """
    Ford-Johnson sorting algorithm.

    Examples
    --------
        >>> fj_sort = SortFordJohnson(compute_history=False)
        >>> perm = np.array([14, 2, 0, 10, 13, 5, 18, 19, 7, 12, 6, 15, 16, 1, 3, 4, 8, 17, 11, 9])
        >>> fj_sort(perm).n_comparisons_
        60
        >>> fj_sort.perm_  # doctest: +NORMALIZE_WHITESPACE
        array([14,  2,  0, 10, 13,  5, 18, 19,  7, 12,  6, 15, 16,  1,  3,  4,  8,  17, 11,  9])
        >>> fj_sort.sorted_list_  # doctest: +NORMALIZE_WHITESPACE
        array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    """

    __name__ = 'ford_johnson'

    def __init__(self, compute_history=False):
        super().__init__(compute_history=compute_history)
        self.sorted_indices_ = None

    def _initialize_algo_aux(self):
        pass

    def _call_aux(self):
        self.sorted_indices_ = np.array(_ford_johnson_sorting(np.arange(self.n_), lt=self.test_i_lt_j))

    def distance_to_sorted_array(self):
        return None  # TODO: implement distance to sorted array

    @property
    def sorted_list_(self):
        return self.perm_[self.sorted_indices_]


def _binary_search_insertion(sorted_list, item, lt):
    """

    Parameters
    ----------
    sorted_list: class `list`
                 A sorted list
    item: class: `int`
                 An element to insert in the sorted list
    lt: callable
        lt(x, y) is the test used to determine whether element x is lower than y.

    Returns
    -------
    :class:`tuple`
        The couple of sorted list with the inserted item, and its position

    Examples
    --------
        >>> nc = 0
        >>> def my_lt(x, y):
        ...     global nc
        ...     nc += 1
        ...     return x < y
        >>> _binary_search_insertion([1, 12, 45, 51, 69, 99], 42, my_lt)
        ([1, 12, 42, 45, 51, 69, 99], 2)
        >>> nc
        3
    """
    left = 0
    right = len(sorted_list) - 1
    while left <= right:
        middle = (left + right) // 2
        if left == right:
            if lt(sorted_list[middle], item):
                left = middle + 1
            break
        else:
            if lt(sorted_list[middle], item):
                left = middle + 1
            else:
                right = middle - 1
    sorted_list.insert(left, item)
    return sorted_list, left


def _give_the_right_order(n):  # In ford_johnson_sorting, always need to put len(collection)-1
    """

    Parameters
    ----------
    n: class: `int`
             The size of the collection + 1. Must be >= 0 (i.e. the collection must have at least one element).
    Returns
    -------
    :class: `list`
            A list that gives the right order of insertion for the last step of ford johnson

    Examples
    --------
        >>> _give_the_right_order(7)
        [5, 6, 3, 4, 0, 1, 2]
        >>> _give_the_right_order(0)
        []
    """
    my_list = []
    k = 1  # number of the set
    i = 0
    position_k = 0  # insertion position when elt belongs to set k
    while i < n:
        cpt = 0
        if k % 2 == 0:
            while (cpt < (2**k + (-2-(-2)**k)//3)) and i < n:
                my_list.insert(position_k, n - 1 - i)
                i += 1
                cpt += 1
            position_k += (2**k + (-2-(-2)**k)//3)
        else:
            while (cpt < (2**k - (-2-(-2)**k)//3)) and i < n:
                my_list.insert(position_k, n - 1 - i)
                i += 1
                cpt += 1
            position_k += (2**k - (-2-(-2)**k)//3)
        k += 1
    return my_list


def _update_indices(position, nb_iter, positions_of_insertion):
    """

    Parameters
    ----------
    position: class: `int`
                      The position of the last element inserted (during last step)
    nb_iter: class: `int`
                     The number of elements inserted so far (during the last step)
    positions_of_insertion: class: `list`
            The list of positions of insertions

    Returns
    -------
    :class: `list`
            The list of updated positions of insertion

    Examples
    --------

        >>> my_position = 4
        >>> my_nb_iter = 1
        >>> my_positions_of_insertion = [1, 4, 5, 0, 9, 3]
        >>> _update_indices(my_position, my_nb_iter, my_positions_of_insertion)
        [1, 5, 6, 0, 10, 3]
    """
    for k in range(nb_iter, len(positions_of_insertion)):
        if positions_of_insertion[k] >= position:
            positions_of_insertion[k] += 1
    return positions_of_insertion


def _insert_y(sorted_pairs, result, lt):
    """

    Parameters
    ----------
    sorted_pairs: class: `list of lists`
            The list of sorted pairs. Must be non-empty.
    result: class: `list`
            The sorted list associated to the first items
    lt: callable
        lt(x, y) is the test used to determine whether element x is lower than y.

    Returns
    -------
    :class: `list`
            The list of sorted item after inserting all the second items

    Examples
    --------

        >>> nc = 0
        >>> def my_lt(x, y):
        ...     global nc
        ...     nc += 1
        ...     return x < y
        >>> res =[40, 100, 200, 459, 600, 999, 1000, 2000]
        >>> my_sorted_pairs =[[40, 75], [100, 343], [200, 201], [459, 568], [600, 3000], [1000, 2000]]
        >>> _insert_y(my_sorted_pairs, res, my_lt)
        [40, 75, 100, 200, 201, 343, 459, 568, 600, 999, 1000, 2000, 3000]
        >>> nc
        13

        >>> res =[40, 75]
        >>> my_sorted_pairs =[[40, 75]]
        >>> _insert_y(my_sorted_pairs, res, my_lt)
        [40, 75]
    """
    n = len(sorted_pairs)
    order = _give_the_right_order(n - 1)  # this list will never move
    position = order[:]  # this list will be updated after each insertion of y
    for i in range(len(order)):
        # Insert y in the appropriate sublist, and extract its sub-position:
        (new_list, pos) = _binary_search_insertion(result[position[i] + 1:], sorted_pairs[order[i]][1], lt)
        # Concatenate the two lists:
        result = result[:position[i]+1] + new_list
        # Update the new positions of insertion, don't forget to update according to position in result,
        # and not sub-position of y:
        _update_indices(pos + 1 + position[i], i, position)
    return result


def _create_pairs(elements_to_sort, lt):
    """

    Parameters
    ----------
    elements_to_sort: class `list`
              The list of elements to sort
    lt: callable
        lt(x, y) is the test used to determine whether element x is lower than y.

    Returns
    -------
    : class `tuple`
             A couple with the sorted pairs, and the last item if there is one (otherwise -1)

    Examples
    --------
        >>> my_elements_to_sort = [1, 0, 3, 4, 5, 6]
        >>> _create_pairs(my_elements_to_sort, lt=lambda x, y: x < y)
        ([[0, 1], [3, 4], [5, 6]], -1)
    """
    two_paired_list = []
    has_last_odd_item = False
    last_item = None
    for i in range(0, len(elements_to_sort), 2):
        if i == len(elements_to_sort) - 1:
            has_last_odd_item = True
            last_item = elements_to_sort[len(elements_to_sort) - 1]
        else:
            if lt(elements_to_sort[i], elements_to_sort[i + 1]):
                two_paired_list.append([elements_to_sort[i], elements_to_sort[i + 1]])
            else:
                two_paired_list.append([elements_to_sort[i + 1], elements_to_sort[i]])
    if has_last_odd_item:
        return two_paired_list, last_item
    else:
        return two_paired_list, -1


def _ford_johnson_sorting(collection, lt=None):
    """
    Ford-Johnson sorting algorithm

    Parameters
    ----------
    collection: class: `list`
                A list to sort
    lt: callable
        lt(x, y) is the test used to determine whether element x is lower than y.
        Default: operator "<".

    Returns
    -------
    :class: `list`
            The sorted list

    Examples
    --------
        >>> my_collection = [14, 2, 0, 10, 13, 5, 18, 19, 7, 12, 6, 15, 16, 1, 3, 4, 8, 17, 11, 9]
        >>> _ford_johnson_sorting(my_collection)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]

    Use the parameter `lt` to add a counter of comparisons:

        >>> nc = 0
        >>> def my_lt(my_x, my_y):
        ...     global nc
        ...     nc += 1
        ...     return my_x < my_y
        >>> my_collection = [14, 2, 0, 10, 13, 5, 18, 19, 7, 12, 6, 15, 16, 1, 3, 4, 8, 17, 11, 9]
        >>> _ford_johnson_sorting(my_collection, my_lt)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        >>> nc
        60

    Misc particular cases:

        >>> _ford_johnson_sorting([])
        []
        >>> _ford_johnson_sorting([42])
        [42]
        >>> _ford_johnson_sorting([42, 51])
        [42, 51]
        >>> _ford_johnson_sorting([51, 42])
        [42, 51]
        >>> _ford_johnson_sorting([51, 42, 12])
        [12, 42, 51]
    """
    if lt is None:
        def lt(x, y):
            return x < y

    if len(collection) <= 1:
        return collection
    (pairs, last_elt) = _create_pairs(collection, lt)
    m = len(pairs)
    pairs_x = [pairs[i][0] for i in range(m)]
    pairs_y = [pairs[i][1] for i in range(m)]
    new_pairs_x = _ford_johnson_sorting(pairs_x, lt)
    new_pairs_y = [pairs_y[pairs_x.index(new_pairs_x[i])] for i in range(m)]
    pairs = [[new_pairs_x[i], new_pairs_y[i]] for i in range(m)]
    result = new_pairs_x
    result.append(pairs[m-1][1])
    if last_elt >= 0:
        (result, pos) = _binary_search_insertion(result, last_elt, lt)
    return _insert_y(pairs, result, lt)
