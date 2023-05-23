from corsort import WrapSortScorer, jit_scorer_rho, SortQuick


# noinspection PyTypeChecker
def test_wrap_sort_scorer():
    """
    To cover 'perm = np.array(perm) ' + compute_history:

        >>> my_sort = SortQuick(compute_history=False)
        >>> jit_sort = WrapSortScorer(scorer=jit_scorer_rho, sort=my_sort, compute_history=True)
        >>> jit_sort([2, 3, 1]).n_comparisons_
        2
    """
    pass
