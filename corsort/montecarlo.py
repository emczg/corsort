import numpy as np
from multiprocess.pool import Pool
from collections import defaultdict
from functools import partial
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

    >>> from corsort import SortQuick, corsort_borda_fast, entropy_bound
    >>> my_nt = 100
    >>> my_n = 10
    >>> np.random.seed(42)
    >>> quicksort = SortQuick(compute_history=False)
    >>> quicksort.__name__ = 'quicksort'
    >>> my_sort_list = [quicksort, corsort_borda_fast]
    >>> my_n_list = [10, 15]

    With evaluate corsort and quicksort using a Pool:

    >>> with Pool() as p:
    ...     my_res = evaluate(my_sort_list, my_n_list, nt=my_nt, pool=p)
    Evaluate quicksort for n = 10
    Evaluate corsort_borda_fast for n = 10
    Evaluate quicksort for n = 15
    Evaluate corsort_borda_fast for n = 15
    >>> print_res(my_res)
    n=10, quicksort: mean=24.05, std=3.52
    n=15, quicksort: mean=46.72, std=6.90
    n=10, corsort_borda_fast: mean=22.11, std=0.87
    n=15, corsort_borda_fast: mean=40.59, std=1.33

    Same without the pool:

    >>> np.random.seed(42)
    >>> my_res = evaluate(my_sort_list, my_n_list, nt=my_nt)
    Evaluate quicksort for n = 10
    Evaluate corsort_borda_fast for n = 10
    Evaluate quicksort for n = 15
    Evaluate corsort_borda_fast for n = 15
    >>> print_res(my_res)
    n=10, quicksort: mean=24.05, std=3.52
    n=15, quicksort: mean=46.72, std=6.90
    n=10, corsort_borda_fast: mean=22.11, std=0.87
    n=15, corsort_borda_fast: mean=40.59, std=1.33

    Bound (loose, not exact):

    >>> print("\\n".join(f"Bound for n={my_n}: {entropy_bound(my_n):.2f}" for my_n in my_n_list))
    Bound for n=10: 22.11
    Bound for n=15: 40.87
    """
    res = defaultdict(dict)
    for n in n_list:
        for sort in sort_list:
            print(f"Evaluate {sort.__name__} for n = {n}")
            convergence_times = np.zeros(nt, dtype=int)
            distances = []
            if pool is not None:
                for k, cd in enumerate(pool.imap_unordered(sort,
                                                           tqdm([np.random.permutation(n)
                                                                 for _ in range(nt)]))):
                    convergence_times[k] = cd[0]
                    distances.append(cd[1])
            else:
                for k in tqdm(range(nt)):
                    cd = sort(np.random.permutation(n))
                    convergence_times[k] = cd[0]
                    distances.append(cd[1])
            max_d = max(len(d) for d in distances)
            dist_array = np.zeros((nt, max_d), dtype=int)
            for i, dist in enumerate(distances):
                dist_array[i, :len(dist)] = dist
            res[sort.__name__][n] = {'time': convergence_times, 'distance': dist_array}
    return res


def evaluate_convergence(sort_list, n, nt, pool=None):
    res = dict()
    for raw_sort in sort_list:
        name = raw_sort.__name__
        sort = partial(raw_sort, compute_history=True)
        print(f"Evaluate convergence of {name} for n = {n}")
        distances = []
        if pool is not None:
            for k, cd in enumerate(pool.imap_unordered(sort,
                                                       tqdm([np.random.permutation(n)
                                                             for _ in range(nt)]))):
                distances.append(cd[1])
        else:
            for _ in tqdm(range(nt)):
                cd = sort(np.random.permutation(n))
                distances.append(cd[1])
        max_d = max(len(d) for d in distances)
        dist_array = np.zeros((nt, max_d), dtype=int)
        for i, dist in enumerate(distances):
            dist_array[i, :len(dist)] = dist
        res[name] = dist_array
    return res


def evaluate_comparisons(sort_list, n_list, nt, pool=None):
    res = defaultdict(dict)
    for n in n_list:
        for sort in sort_list:
            print(f"Evaluate comparisons of {sort.__name__} for n = {n}")
            convergence_times = np.zeros(nt, dtype=int)
            if pool is not None:
                for k, cd in enumerate(pool.imap_unordered(sort,
                                                           tqdm([np.random.permutation(n)
                                                                 for _ in range(nt)]))):
                    convergence_times[k] = cd[0]
            else:
                for k in tqdm(range(nt)):
                    cd = sort(np.random.permutation(n))
                    convergence_times[k] = cd[0]
            res[sort.__name__][n] = convergence_times
    return res
