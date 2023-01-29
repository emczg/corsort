import numpy as np
from multiprocess.pool import Pool
from tqdm import tqdm


def print_res(sort_list, n_list, means, stds):
    for j, n in enumerate(n_list):
        for i, sort in enumerate(sort_list):
            print(f"n={n}, {sort.__name__}: mean={means[i, j]:.2f}, std={stds[i, j]:.2f}")


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
    ...     means, stds = evaluate(sort_list, n_list, nt=n_t, pool=p)
    Evaluate quicksort for n = 10
    Evaluate corsort for n = 10
    Evaluate quicksort for n = 15
    Evaluate corsort for n = 15
    >>> print_res(sort_list, n_list, means, stds)
    n=10, quicksort: mean=24.05, std=3.52
    n=10, corsort: mean=22.12, std=0.89
    n=15, quicksort: mean=46.72, std=6.90
    n=15, corsort: mean=40.94, std=1.34

    Same without the pool:

    >>> np.random.seed(42)
    >>> means, stds = evaluate(sort_list, n_list, nt=n_t)
    Evaluate quicksort for n = 10
    Evaluate corsort for n = 10
    Evaluate quicksort for n = 15
    Evaluate corsort for n = 15
    >>> print_res(sort_list, n_list, means, stds)
    n=10, quicksort: mean=24.05, std=3.52
    n=10, corsort: mean=22.12, std=0.89
    n=15, quicksort: mean=46.72, std=6.90
    n=15, corsort: mean=40.94, std=1.34

    Bound (loose, not exact):

    >>> print("\\n".join(f"Bound for n={n}: {entropy_bound(n):.2f}" for n in n_list))
    Bound for n=10: 22.11
    Bound for n=15: 40.87
    """
    means = np.zeros((len(sort_list), len(n_list)))
    stds = np.zeros((len(sort_list), len(n_list)))
    for j, n in enumerate(n_list):
        for i, sort in enumerate(sort_list):
            print(f"Evaluate {sort.__name__} for n = {n}")
            convs = np.zeros(nt, dtype=int)
            if pool is not None:
                for k, cd in enumerate(pool.imap_unordered(sort,
                                                           tqdm([np.random.permutation(n)
                                                                 for _ in range(nt)]))):
                    convs[k] = cd[0]
            else:
                for k in tqdm(range(nt)):
                    convs[k] = sort(np.random.permutation(n))[0]
            means[i, j], stds[i, j] = np.mean(convs), np.std(convs)
    return means, stds
