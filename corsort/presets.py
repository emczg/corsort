from matplotlib import pylab as plt

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

sorts = ['corsort_delta_max_rho', 'mergesort_bfs', 'ford_johnson_rho',
         'mergesort_dfs_rho', 'mergesort_dfs', 'mergesort_bfs_rho',
         'quicksort', 'heapsort']

color_dict = {k: v for k, v in zip(sorts, colors)}
color_dict['ford_johnson'] = color_dict['ford_johnson_rho']


def auto_colors(sort_list):
    """
    Maps a list of sorts to standard pyplot colors.

    Parameters
    ----------
    sort_list: :class:`list`

    Returns
    -------
    :class:`dict`

    Examples
    --------

    >>> from corsort.sort_ford_johnson import SortFordJohnson
    >>> from corsort.sort_merge_bfs import SortMergeBfs
    >>> from corsort.sort_merge_dfs import SortMergeDfs
    >>> from corsort.sort_quick import SortQuick
    >>> s_list = [SortFordJohnson(), SortQuick, SortMergeBfs(), SortMergeDfs]
    >>> auto_colors(s_list)
    {'ford_johnson': '#1f77b4', 'SortQuick': '#ff7f0e', 'mergesort_bfs': '#2ca02c', 'SortMergeDfs': '#d62728'}
    """
    return {k.__name__: v for k, v in zip(sort_list, colors)}
