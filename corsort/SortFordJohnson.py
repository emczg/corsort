import numpy as np
from corsort.Sort import Sort


class SortFordJohnson(Sort):
    """
    Ford-Johnson sorting algorithm.

    Examples
    --------
        >>> fj_sort = SortFordJohnson(compute_history=False)
        >>> L = [14,2,0,10,13,5,18,19,7,12,6,15,16,1,3,4,8,17,11,9]
        >>> fj_sort(L).n_comparisons_
        60
        >>> fj_sort.perm_  # doctest: +NORMALIZE_WHITESPACE
        array([14,  2,  0, 10, 13,  5, 18, 19,  7, 12,  6, 15, 16,  1,  3,  4,  8,  17, 11,  9])
        >>> fj_sort.sorted_list_  # doctest: +NORMALIZE_WHITESPACE
        array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
    """

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
        >>> def lt(x, y):
        ...     global nc
        ...     nc += 1
        ...     return x < y
        >>> _binary_search_insertion([1, 12, 45, 51, 69, 99], 42, lt)
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
             The size of the collection + 1
    Returns
    -------
    :class: `list`
            A list that gives the right order of insertion for the last step of ford johnson

    Examples
    --------
        >>> _give_the_right_order(7)
        [5, 6, 3, 4, 0, 1, 2]
    """
    if n == 0:
        return[0]
    L = []
    k = 1  # number of the set
    i = 0
    position_k = 0  # insertion position when elt belongs to set k
    while i < n:
        cpt=0
        if k % 2 == 0:
            while (cpt < (2**k + (-2-(-2)**k)//3)) and i < n:
                L.insert(position_k, n-1-i)
                i += 1
                cpt += 1
            position_k += (2**k + (-2-(-2)**k)//3)
        else:
            while (cpt < (2**k - (-2-(-2)**k)//3)) and i < n:
                L.insert(position_k, n-1-i)
                i += 1
                cpt += 1
            position_k += (2**k - (-2-(-2)**k)//3)
        k += 1
    return L


def _update_indices(position, nb_iter, L):
    """

    Parameters
    ----------
    position: class: `int`
                      The position of the last element inserted (during last step)
    nb_iter: class: `int`
                     The number of elements inserted so far (during the last step)
    L: class: `list`
            The list of positions of insertions

    Returns
    -------
    :class: `list`
            The list of updated positions of insertion

    Examples
    --------

        >>> position = 4
        >>> nb_iter = 1
        >>> L = [1,4,5,0,9,3]
        >>> _update_indices(position, nb_iter, L)
        [1, 5, 6, 0, 10, 3]
    """
    for k in range(nb_iter, len(L)):
        if L[k] >= position:
            L[k] += 1
    return L


def _insert_y(M, result, lt):
    """

    Parameters
    ----------
    M: class: `list of lists`
            The list of sorted pairs
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
        >>> def lt(x, y):
        ...     global nc
        ...     nc += 1
        ...     return x < y
        >>> res=[40,100,200,459,600,999,1000,2000]
        >>> MM=[[40,75],[100,343],[200,201],[459,568],[600,3000],[1000,2000]]
        >>> _insert_y(MM,res,lt)
        [40, 75, 100, 200, 201, 343, 459, 568, 600, 999, 1000, 2000, 3000]
        >>> nc
        13

    """
    n = len(M)
    cpt = 0

    order = _give_the_right_order(n - 1)  # this list will never move
    position = _give_the_right_order(n - 1)  # this list will be updated after each insertion of y
    for i in range(len(order)):
        (new_list, pos) = _binary_search_insertion(result[position[i] + 1:], M[order[i]][1], lt)  # insert y in the
        # appropriate sublist, and extract its sub-position
        result = result[:position[i]+1] + new_list  # concatenate the two lists
        _update_indices(pos + 1 + position[i], cpt, position) # update the new positions of insertion, don't forget to
        # update according to position in result, and not sub-position of y
        cpt += 1  # update the number of inserted elements
    return result


def _create_pairs(L, lt):
    """

    Parameters
    ----------
    L: class `list`
              The list of elements to sort
    lt: callable
        lt(x, y) is the test used to determine whether element x is lower than y.

    Returns
    -------
    : class `tuple`
             A couple with the sorted pairs, and the last item if there is one (otherwise -1)

    Examples
    --------
        >>> L = [1,0,3,4,5,6]
        >>> _create_pairs(L, lt=lambda x, y: x < y)
        ([[0, 1], [3, 4], [5, 6]], -1)
    """
    two_paired_list = []
    has_last_odd_item = False
    for i in range(0, len(L), 2):
        if i == len(L) - 1:
            has_last_odd_item = True
            last_item = L[len(L)-1]
        else:
            if lt(L[i], L[i + 1]):
                two_paired_list.append([L[i], L[i + 1]])
            else:
                two_paired_list.append([L[i + 1], L[i]])
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

        >>> nc = 0
        >>> def lt(x, y):
        ...     global nc
        ...     nc += 1
        ...     return x < y
        >>> L = [14,2,0,10,13,5,18,19,7,12,6,15,16,1,3,4,8,17,11,9]
        >>> _ford_johnson_sorting(L, lt)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        >>> nc
        60

    """
    if lt is None:
        def lt(x, y):
            return x < y

    if len(collection) == 1:
        L = collection
    elif len(collection) == 2:
        if lt(collection[0], collection[1]):
            L = [collection[0], collection[1]]
        else:
            L = [collection[1], collection[0]]
    elif len(collection) == 3:
        (M, last_elt) = _create_pairs(collection, lt)
        L = [M[0][0], M[0][1]]
        L, pos = _binary_search_insertion(L, last_elt, lt)
    else:
        (M, last_elt) = _create_pairs(collection, lt)
        m = len(M)
        M_x = [M[i][0] for i in range(m)]
        M_y = [M[i][1] for i in range(m)]
        new_M_x = _ford_johnson_sorting(M_x, lt)
        new_M_y = [M_y[M_x.index(new_M_x[i])] for i in range(m)]
        M = [[new_M_x[i], new_M_y[i]] for i in range(m)]
        L = new_M_x
        L.append(M[m-1][1])
        if last_elt >= 0:
            (L, pos) = _binary_search_insertion(L, last_elt, lt)
        L = _insert_y(M, L, lt)
    return L

