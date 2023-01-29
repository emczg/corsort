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

    pool: :class:`~mulyiprocess.pool.Pool`, optional.
        Use parallelism.
    Returns
    -------

    Examples
    --------

    >>> from corsort import quicksort, corsort, entropy_bound
    >>> n_t = 100
    >>> n = 10
    >>> np.random.seed(42)
    >>> sort_list = [quicksort, corsort]
    >>> n_list = [10, 15]

    With evaluate corsort and quicksort using a Pool:

    >>> with Pool() as p:
    ...     res = evaluate(sort_list, n_list, nt=n_t, pool=p)
    Evaluate quicksort for n = 10
    Evaluate corsort for n = 10
    Evaluate quicksort for n = 15
    Evaluate corsort for n = 15
    >>> print_res(res)
    n=10, quicksort: mean=24.05, std=3.52
    n=15, quicksort: mean=46.72, std=6.90
    n=10, corsort: mean=22.12, std=0.89
    n=15, corsort: mean=40.94, std=1.34

    Same without the pool:

    >>> np.random.seed(42)
    >>> res = evaluate(sort_list, n_list, nt=n_t)
    Evaluate quicksort for n = 10
    Evaluate corsort for n = 10
    Evaluate quicksort for n = 15
    Evaluate corsort for n = 15
    >>> print_res(res)
    n=10, quicksort: mean=24.05, std=3.52
    n=15, quicksort: mean=46.72, std=6.90
    n=10, corsort: mean=22.12, std=0.89
    n=15, corsort: mean=40.94, std=1.34

    Bound (loose, not exact):

    >>> print("\\n".join(f"Bound for n={n}: {entropy_bound(n):.2f}" for n in n_list))
    Bound for n=10: 22.11
    Bound for n=15: 40.87
    """
    res = defaultdict(dict)
    for n in n_list:
        for sort in sort_list:
            print(f"Evaluate comparisons of {sort.__name__} for n = {n}")
            convs = np.zeros(nt, dtype=int)
            distances = []
            if pool is not None:
                for k, cd in enumerate(pool.imap_unordered(sort,
                                                           tqdm([np.random.permutation(n)
                                                                 for _ in range(nt)]))):
                    convs[k] = cd[0]
                    distances.append(cd[1])
            else:
                for k in tqdm(range(nt)):
                    cd = sort(np.random.permutation(n))
                    convs[k] = cd[0]
                    distances.append(cd[1])
            max_d = max(len(d) for d in distances)
            dist_array = np.zeros((nt, max_d), dtype=int)
            for i, dist in enumerate(distances):
                dist_array[i, :len(dist)] = dist
            res[sort.__name__][n] = {'time': convs, 'distance': dist_array}
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
            for k in tqdm(range(nt)):
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
            print(f"Evaluate {sort.__name__} for n = {n}")
            convs = np.zeros(nt, dtype=int)
            if pool is not None:
                for k, cd in enumerate(pool.imap_unordered(sort,
                                                           tqdm([np.random.permutation(n)
                                                                 for _ in range(nt)]))):
                    convs[k] = cd[0]
            else:
                for k in tqdm(range(nt)):
                    cd = sort(np.random.permutation(n))
                    convs[k] = cd[0]
            res[sort.__name__][n] = convs
    return res

