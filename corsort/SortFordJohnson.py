import numpy as np
from corsort.Sort import Sort


class SortFordJohnson(Sort):
    """
    Ford-Johnson sorting algorithm.

    Examples
    --------
        >>> fj_sort = SortFordJohnson(compute_history=False)
        >>> L = [14,2,0,10,13,5,18,19,7,12,6,15,16,1,3,4,8,17,11,9]
        >>> fj_sort(L)
        (60, [])
        >>> fj_sort.perm_  # doctest: +NORMALIZE_WHITESPACE
        array([14,  2,  0, 10, 13,  5, 18, 19,  7, 12,  6, 15, 16,  1,  3,  4,  8,  17, 11,  9])
        >>> fj_sort.sorted_list_  # doctest: +NORMALIZE_WHITESPACE
        array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
        >>> fj_sort.n_comparisons_
        60
    """

    def __init__(self, compute_history=False):
        super().__init__(compute_history=compute_history)
        self.sorted_list_ = None

    def _call_aux(self):
        nc = [0]
        self.sorted_list_ = np.array(ford_johnson_sorting(self.perm_, nc=nc))
        self.n_comparisons_ = nc[0]
        self.history_distances_ = []  # TODO: implement history of distance


def binary_search_insertion(sorted_list, item, nc):
    """

    Parameters
    ----------
    sorted_list: class `list`
                 A sorted list
    item: class: `int`
                 An element to insert in the sorted list
    nc: class: `list`
                 A list of one element with the current number of comparisons

    Returns
    -------
    :class:`tuple`
        The couple of sorted list with the inserted item, and its position

    Examples
    --------
        >>> nc = [0]
        >>> binary_search_insertion([1, 12, 45, 51, 69, 99], 42, nc)
        ([1, 12, 42, 45, 51, 69, 99], 2)
        >>> nc[0]
        3
    """
    left = 0
    right = len(sorted_list) - 1
    while left <= right:
        middle = (left + right) // 2
        if left == right:
            nc[0] += 1
            if sorted_list[middle] < item:
                left = middle + 1
            break
        else:
            nc[0] += 1
            if sorted_list[middle] < item:
                left = middle + 1
            else:
                right = middle - 1
    sorted_list.insert(left, item)
    return sorted_list, left


def give_the_right_order(n):  # In ford_johnson_sorting, always need to put len(collection)-1
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
        >>> n = 7
        >>> give_the_right_order(n)
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


def update_indices(position, nb_iter, L):
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
        >>> update_indices(position, nb_iter, L)
        [1, 5, 6, 0, 10, 3]
    """
    for k in range(nb_iter, len(L)):
        if L[k] >= position:
            L[k] += 1
    return L


def insert_y(M, result, nc):
    """

    Parameters
    ----------
    M: class: `list of lists`
            The list of sorted pairs
    result: class: `list`
            The sorted list associated to the first items
    nc: class `list`
               A list of one element with the current number of comparisons

    Returns
    -------
    :class: `list`
            The list of sorted item after inserting all the second items

    Examples
    --------

        >>> res=[40,100,200,459,600,999,1000,2000]
        >>> MM=[[40,75],[100,343],[200,201],[459,568],[600,3000],[1000,2000]]
        >>> nc=[0]
        >>> insert_y(MM,res,nc)
        [40, 75, 100, 200, 201, 343, 459, 568, 600, 999, 1000, 2000, 3000]
        >>> nc[0]
        13

    """
    n = len(M)
    cpt = 0

    order = give_the_right_order(n-1)  # this list will never move
    position = give_the_right_order(n-1)  # this list will be updated after each insertion of y
    pos = position[0]
    for i in range(len(order)):
        (new_list, pos) = binary_search_insertion(result[position[i]+1:], M[order[i]][1], nc)  # insert y in the
        # appropriate sublist, and extract its sub-position
        result = result[:position[i]+1] + new_list  # concatenate the two lists
        update_indices(pos+1+position[i],cpt,position) # update the new positions of insertion, don't forget to
        # update according to position in result, and not sub-position of y
        cpt += 1  # update the number of inserted elements
    return result


def create_pairs(L, nc):
    """

    Parameters
    ----------
    L: class `list`
              The list of elements to sort
    nc: class `list`
              A list of one element with the current number of comparisons

    Returns
    -------
    : class `tuple`
             A couple with the sorted pairs, and the last item if there is one (otherwise -1)

    Examples
    --------
        >>> L = [1,0,3,4,5,6]
        >>> create_pairs(L,[0])
        ([[0, 1], [3, 4], [5, 6]], -1)
    """
    two_paired_list = []
    has_last_odd_item = False
    for i in range(0, len(L), 2):
        if i == len(L) - 1:
            has_last_odd_item = True
            last_item = L[len(L)-1]
        else:
            nc[0] += 1
            if L[i] < L[i + 1]:
                two_paired_list.append([L[i], L[i + 1]])
            else:
                two_paired_list.append([L[i + 1], L[i]])
    if has_last_odd_item:
        return two_paired_list, last_item
    else:
        return two_paired_list, -1


def ford_johnson_sorting(collection, nc):
    """
    Ford-Johnson sorting algorithm

    Parameters
    ----------
    collection: class: `list`
                A list to sort
    nc: class: `list`
                A list of one element with the current number of comparisons

    Returns
    -------
    :class: `list`
            The sorted list

    Examples
    --------

        >>> L = [14,2,0,10,13,5,18,19,7,12,6,15,16,1,3,4,8,17,11,9]
        >>> nc=[0]
        >>> ford_johnson_sorting(L, nc)
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        >>> nc[0]
        60

    """
    L = []
    if len(collection) == 1:
        L = collection
    elif len(collection) == 2:
        nc[0] += 1
        if collection[0] < collection[1]:
            L = [collection[0], collection[1]]
        else:
            L = [collection[1], collection[0]]
    elif len(collection) == 3:
        (M, last_elt) = create_pairs(collection, nc)
        L = [M[0][0], M[0][1]]
        L, pos = binary_search_insertion(L, last_elt, nc)
    else:
        (M, last_elt) = create_pairs(collection, nc)
        m = len(M)
        M_x = [M[i][0] for i in range(m)]
        M_y = [M[i][1] for i in range(m)]
        new_M_x = ford_johnson_sorting(M_x, nc)
        new_M_y = [M_y[M_x.index(new_M_x[i])] for i in range(m)]
        M = [[new_M_x[i], new_M_y[i]] for i in range(m)]
        L = new_M_x
        L.append(M[m-1][1])
        if last_elt >= 0:
            (L, pos) = binary_search_insertion(L, last_elt, nc)
        L = insert_y(M, L, nc)
    return L

