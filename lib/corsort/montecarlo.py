import numpy as np
from multiprocess.pool import Pool
from collections import defaultdict
from tqdm import tqdm


def print_res(res):
    for name, di in res.items():
        for n, v in di.items():
            t = v['time']
            m = np.mean(t)
            s = np.std(t)
            print(f"n={n}, {name}: mean={m:.2f}, std={s:.2f}")


def evaluate(sort_list, n_list, nt, pool=None):
    """
    Run a sim.

    Parameters
    ----------
    sort_list: :class:`list`
        List of sorting algorithms (cf. examples).
    n_list: :class:`list`
        List of sizes for the tested lists.
    nt: :class:`int`
        Number of samples.
    pool: :class:`~multiprocess.pool.Pool`, optional.
        Use parallelism.

    Returns
    -------

    Examples
    --------

    >>> from corsort import SortQuick, WrapFullJit, entropy_bound, jit_corsort_borda
    >>> my_nt = 100
    >>> np.random.seed(42)
    >>> my_sort_list = [SortQuick(), WrapFullJit(jit_corsort_borda)]
    >>> my_n_list = [10, 15]

    Evaluate corsort and quicksort using a Pool:

    >>> with Pool() as p:
    ...     my_res = evaluate(my_sort_list, my_n_list, nt=my_nt, pool=p)
    Evaluate quicksort for n = 10
    Evaluate corsort_borda for n = 10
    Evaluate quicksort for n = 15
    Evaluate corsort_borda for n = 15
    >>> print_res(my_res)
    n=10, quicksort: mean=24.05, std=3.52
    n=15, quicksort: mean=46.72, std=6.90
    n=10, corsort_borda: mean=22.11, std=0.87
    n=15, corsort_borda: mean=40.59, std=1.33

    Same without the pool:

    >>> np.random.seed(42)
    >>> my_res = evaluate(my_sort_list, my_n_list, nt=my_nt)
    Evaluate quicksort for n = 10
    Evaluate corsort_borda for n = 10
    Evaluate quicksort for n = 15
    Evaluate corsort_borda for n = 15
    >>> print_res(my_res)
    n=10, quicksort: mean=24.05, std=3.52
    n=15, quicksort: mean=46.72, std=6.90
    n=10, corsort_borda: mean=22.11, std=0.87
    n=15, corsort_borda: mean=40.59, std=1.33

    Bound (loose, not exact):

    >>> print("\\n".join(f"Bound for n={my_n}: {entropy_bound(my_n):.2f}" for my_n in my_n_list))
    Bound for n=10: 21.78
    Bound for n=15: 40.24
    """
    res = defaultdict(dict)
    for n in n_list:
        for sort in sort_list:
            print(f"Evaluate {sort.__name__} for n = {n}")
            convergence_times = np.zeros(nt, dtype=int)
            distances = []
            if pool is not None:
                for k, the_sort in enumerate(pool.imap_unordered(sort,
                                                                 tqdm([np.random.permutation(n)
                                                                       for _ in range(nt)]))):
                    convergence_times[k] = the_sort.n_comparisons_
                    distances.append(the_sort.history_distances_)
            else:
                for k in tqdm(range(nt)):
                    _ = sort(np.random.permutation(n))
                    convergence_times[k] = sort.n_comparisons_
                    distances.append(sort.history_distances_)
            max_d = max(len(d) for d in distances)
            dist_array = np.zeros((nt, max_d), dtype=int)
            for i, dist in enumerate(distances):
                dist_array[i, :len(dist)] = dist
            res[sort.__name__][n] = {'time': convergence_times, 'distance': dist_array}
    return res


def evaluate_convergence(sort_list, n, nt, pool=None):
    """
    Performance profile.

    Parameters
    ----------
    sort_list: :class:`list`
        List of sorting algorithms (cf. examples).
    n: int
        Size for the tested lists.
    nt: :class:`int`
        Number of samples.
    pool: :class:`~multiprocess.pool.Pool`, optional.
        Use parallelism.

    Returns
    -------
    dict
        Key: name of the sorting algorithm. Value: a ndarray of `nt` rows. Each row gives
        the history of distance to the sorted list.

    Examples
    --------
    >>> from corsort import SortQuick, WrapFullJit, entropy_bound, jit_corsort_borda
    >>> my_nt = 100
    >>> np.random.seed(42)
    >>> my_sort_list = [SortQuick(), WrapFullJit(jit_corsort_borda)]
    >>> my_n = 10

    Evaluate corsort and quicksort using a Pool:

    >>> with Pool() as p:
    ...     my_res = evaluate_convergence(my_sort_list, my_n, nt=my_nt, pool=p)
    Evaluate convergence of quicksort for n = 10
    Evaluate convergence of corsort_borda for n = 10
    >>> np.round(np.mean(my_res['quicksort'], axis=0), 1)  # doctest: +NORMALIZE_WHITESPACE
    array([22.7, 22.1, 21.5, 20.6, 19.9, 18.8, 17.4, 16.1, 14.1, 12. , 11.6,
           11. , 10.3,  9.3,  8.3,  7.2,  6.2,  5.4,  4.6,  3.8,  3.1,  2.5,
            1.7,  1.3,  1. ,  0.8,  0.6,  0.4,  0.2,  0.2,  0.1,  0. ,  0. ,
            0. ,  0. ,  0. ,  0. ])

    Same without the pool:

    >>> np.random.seed(42)
    >>> my_res = evaluate_convergence(my_sort_list, my_n, nt=my_nt)
    Evaluate convergence of quicksort for n = 10
    Evaluate convergence of corsort_borda for n = 10
    >>> np.round(np.mean(my_res['quicksort'], axis=0), 1)  # doctest: +NORMALIZE_WHITESPACE
    array([22.7, 22.1, 21.5, 20.6, 19.9, 18.8, 17.4, 16.1, 14.1, 12. , 11.6,
           11. , 10.3,  9.3,  8.3,  7.2,  6.2,  5.4,  4.6,  3.8,  3.1,  2.5,
            1.7,  1.3,  1. ,  0.8,  0.6,  0.4,  0.2,  0.2,  0.1,  0. ,  0. ,
            0. ,  0. ,  0. ,  0. ])
    """
    res = dict()
    for sort in sort_list:
        name = sort.__name__
        compute_history_old = sort.compute_history
        sort.compute_history = True
        print(f"Evaluate convergence of {name} for n = {n}")
        distances = []
        if pool is not None:
            for instant in pool.imap_unordered(sort,
                                               tqdm([np.random.permutation(n)
                                                     for _ in range(nt)])):
                distances.append(instant.history_distances_)
        else:
            for _ in tqdm(range(nt)):
                sort(np.random.permutation(n))
                distances.append(sort.history_distances_)
        max_d = max(len(d) for d in distances)
        dist_array = np.zeros((nt, max_d), dtype=int)
        for i, dist in enumerate(distances):
            dist_array[i, :len(dist)] = dist
        res[name] = dist_array
        sort.compute_history = compute_history_old
    return res


def evaluate_comparisons(sort_list, n_list, nt, pool=None):
    """

    Parameters
    ----------
    sort_list
    n_list
    nt
    pool

    Returns
    -------

    Examples
    --------
    >>> from corsort import SortQuick, WrapFullJit, entropy_bound, jit_corsort_borda
    >>> my_nt = 100
    >>> np.random.seed(42)
    >>> my_sort_list = [SortQuick(), WrapFullJit(jit_corsort_borda)]
    >>> my_n_list = [10, 15]

    Evaluate corsort and quicksort using a Pool:

    >>> with Pool() as p:
    ...     my_res = evaluate_comparisons(my_sort_list, my_n_list, nt=my_nt, pool=p)
    Evaluate comparisons of quicksort for n = 10
    Evaluate comparisons of corsort_borda for n = 10
    Evaluate comparisons of quicksort for n = 15
    Evaluate comparisons of corsort_borda for n = 15
    >>> np.round(np.mean(my_res['quicksort'][10]), 1)
    24.0

    Same without the pool:

    >>> np.random.seed(42)
    >>> my_res = evaluate_comparisons(my_sort_list, my_n_list, nt=my_nt)
    Evaluate comparisons of quicksort for n = 10
    Evaluate comparisons of corsort_borda for n = 10
    Evaluate comparisons of quicksort for n = 15
    Evaluate comparisons of corsort_borda for n = 15
    >>> np.round(np.mean(my_res['quicksort'][10]), 1)
    24.0
    """
    res = defaultdict(dict)
    for n in n_list:
        for sort in sort_list:
            print(f"Evaluate comparisons of {sort.__name__} for n = {n}")
            convergence_times = np.zeros(nt, dtype=int)
            if pool is not None:
                for k, instant in enumerate(pool.imap_unordered(sort,
                                                          tqdm([np.random.permutation(n)
                                                                for _ in range(nt)]))):
                    convergence_times[k] = instant.n_comparisons_
            else:
                for k in tqdm(range(nt)):
                    sort(np.random.permutation(n))
                    convergence_times[k] = sort.n_comparisons_
            res[sort.__name__][n] = convergence_times
    return res
