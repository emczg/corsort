from functools import total_ordering
from corsort.quicksort import quicksort


@total_ordering
class _RecordingItem:
    """
    Wrapper for a list item that records when it is compared to another list item.

    Parameters
    ----------
    value: :class:`object`
        Value of the item.
    position: :class:`int`
        Index of the item in some list.
    history_comparisons: :class:`list`
        A list of pairs. A pair (i, j) means that the items of positions i and j are compared.

    Examples
    --------
        >>> lst = [4, 1, 7, 6, 0, 8, 2, 3, 5]
        >>> history_comparisons = []
        >>> recording_lst = [_RecordingItem(x, pos, history_comparisons) for pos, x in enumerate(lst)]
        >>> recording_lst[0] > recording_lst[1]
        True
        >>> recording_lst[3] < recording_lst[2]
        True
        >>> history_comparisons
        [(0, 1), (3, 2)]
    """

    def __init__(self, value, position, history_comparisons):
        self.value = value
        self.position = position
        self.history_comparisons = history_comparisons

    def __eq__(self, other):
        return self.value == other.value

    def __lt__(self, other):
        self.history_comparisons.append((self.position, other.position))
        return self.value < other.value


def record_comparisons(sort, history_comparisons):
    """
    Wrap a sorting function so that it also records a list of comparisons.

    Parameters
    ----------
    sort: callable
        A sorting algorithm.
    history_comparisons
        A list of pairs. A pair (i, j) records a comparison between items of initial positions `i` and `j`
        in the list to be sorted.

    Returns
    -------
    callable
        The same sorting algorithms, but that also records its comparisons.

    Examples
    --------
    Define the list that will record the pairwise comparisons:

        >>> my_history_comparisons = []

    Wrap the sorting function:

        >>> quicksort_and_record = record_comparisons(quicksort, my_history_comparisons)

    Now use the wrapped function, and the history of comparisons will update automatically:

        >>> quicksort_and_record(['c', 'a', 'b'])
        (3, [])
        >>> my_history_comparisons
        [(1, 0), (2, 0), (2, 1)]

        >>> quicksort_and_record(['c', 'a', 'b', 'd'])
        (4, [])
        >>> my_history_comparisons
        [(1, 0), (2, 0), (3, 0), (2, 1)]
    """
    def sort_and_record_comparisons(xs):
        history_comparisons.clear()
        recording_lst = [_RecordingItem(x, pos, history_comparisons) for pos, x in enumerate(xs)]
        return sort(recording_lst)
    return sort_and_record_comparisons
