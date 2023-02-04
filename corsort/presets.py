from matplotlib import pylab as plt

from corsort.sort_ford_johnson import SortFordJohnson
from corsort.sort_merge_bfs import SortMergeBfs
from corsort.sort_merge_dfs import SortMergeDfs
from corsort.sort_quick import SortQuick
from corsort.wrap_full_jit import WrapFullJit
from corsort.wrap_sort_scorer import WrapSortScorer
from corsort.jit_sorts import jit_corsort_borda, jit_corsort_drift_max_spaced


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

sorts = ['corsort_drift_max_spaced', 'mergesort_bfs', 'ford_johnson_spaced',
         'mergesort_dfs_spaced', 'mergesort_dfs', 'mergesort_bfs_spaced', 'corsort_borda',
         'quicksort']

color_dict = {k: v for k, v in zip(sorts, colors)}
color_dict['ford_johnson'] = color_dict['ford_johnson_spaced']

# sorter_dict = {
#     'corsort': {'sorter': WrapFullJit(jit_corsort_drift_max_spaced),
#                 'name': 'Corsort', 'color': colors[0]}
#
# }
#
#
# merge = SortMergeBfs()
# quick = SortQuick()
# fj = SortFordJohnson()
# corquick = WrapFullJit(jit_corsort_borda)
# corsort = WrapFullJit(jit_corsort_drift_max_spaced)
#
# mbfs = SortMergeBfs(compute_history=True)
# mbfs2 = SortMergeBfs()
# mbfss = WrapSortScorer(scorer=scorer_spaced, sort=mbfs2, compute_history=True)
#
# mdfs = SortMergeDfs(compute_history=True)
# mdfs2 = SortMergeDfs()
# mdfss = WrapSortScorer(scorer=scorer_spaced, sort=mdfs2, compute_history=True)
#
# fj = SortFordJohnson(compute_history=False)
# fjs = WrapSortScorer(scorer=scorer_spaced, sort=fj, compute_history=True)
# dms = WrapFullJit(jit_corsort_drift_max_spaced, compute_history=True)
