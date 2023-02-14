from corsort import WrapFullJit, jit_corsort_borda


# noinspection PyTypeChecker
def test_wrap_full_jit():
    """
    To cover 'perm = np.array(perm) '

        >>> sort = WrapFullJit(jit_sort=jit_corsort_borda)
        >>> sort([2, 3, 1]).n_comparisons_
        2
    """
    pass
