from matplotlib import pylab as plt

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

sorts = ['corsort_delta_max_rho', 'mergesort_bottom_up', 'ford_johnson_rho',
         'mergesort_top_down_rho', 'mergesort_top_down', 'mergesort_bottom_up_rho',
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
    >>> from corsort.sort_merge_bottom_up import SortMergeBottomUp
    >>> from corsort.sort_merge_top_down import SortMergeTopDown
    >>> from corsort.sort_quick import SortQuick
    >>> s_list = [SortFordJohnson(), SortQuick(), SortMergeBottomUp(), SortMergeTopDown()]
    >>> auto_colors(s_list)
    {'ford_johnson': '#1f77b4', 'quicksort': '#ff7f0e', 'mergesort_bottom_up': '#2ca02c', 'mergesort_top_down': '#d62728'}
    """
    return {k.__name__: v for k, v in zip(sort_list, colors)}
