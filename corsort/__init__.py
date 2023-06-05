"""Top-level package for Corsort."""

__author__ = """Emma Caizergues"""
__email__ = 'emma.caizergues@gmail.com'
__version__ = '0.1.2'


from corsort.chain_and_y import ChainAndY, linear_extensions
from corsort.corsort import Corsort
from corsort.corsort_borda import CorsortBorda
from corsort.corsort_chain_decomposition_merge_v import CorsortChainDecompositionMergeV
from corsort.corsort_chain_decomposition_merge_x import CorsortChainDecompositionMergeX
from corsort.corsort_delegate import CorsortDelegate
from corsort.corsort_gain import CorsortGain
from corsort.corsort_gain_lexi import CorsortGainLexi
from corsort.distance_to_sorted_array import distance_to_sorted_array
from corsort.entropy_bound import entropy_bound
from corsort.jit_scorers import jit_scorer_rho, jit_scorer_delta
from corsort.jit_sorts import jit_corsort_borda, jit_corsort_delta_max_rho, \
    jit_corsort_delta_sum_rho, jit_corsort_delta_max_delta, jit_corsort_delta_sum_delta, \
    jit_corsort_rho_max_rho, jit_corsort_rho_sum_rho, jit_corsort_rho_max_delta, \
    jit_corsort_rho_sum_delta, heapify, jit_heapsort
from corsort.merge import merge
from corsort.montecarlo import print_res, evaluate, evaluate_convergence, evaluate_comparisons
from corsort.multi_merge import multi_merge
from corsort.partition import partition
from corsort.presets import colors, sorts, color_dict, auto_colors
from corsort.print_order_as_letters import print_order_as_letters
from corsort.scorers import scorer_delta, scorer_rho
from corsort.sort import Sort
from corsort.sort_asort_quickselect import SortAsortQuickselect
from corsort.sort_binary_insertion import SortBinaryInsertion
from corsort.sort_ford_johnson import SortFordJohnson
from corsort.sort_largest_interval import SortLargestInterval
from corsort.sort_merge_bottom_up import SortMergeBottomUp
from corsort.sort_merge_top_down import SortMergeTopDown
from corsort.sort_multizip import SortMultizip
from corsort.sort_quick import SortQuick
from corsort.sort_shell import SortShell
from corsort.split_pointer_lists import split_pointer_lists
from corsort.util_chains import longest_chain_starting_at, longest_chain, greedy_chain_decomposition
from corsort.util_latex import print_corsort_execution
from corsort.wrap_full_jit import WrapFullJit, JitCorsortBorda, JitHeapsort, \
    JitCorsortDeltaMaxDelta, JitCorsortDeltaMaxRho, JitCorsortDeltaSumDelta, JitCorsortDeltaSumRho, \
    JitCorsortRhoMaxDelta, JitCorsortRhoMaxRho, JitCorsortRhoSumDelta, JitCorsortRhoSumRho
from corsort.wrap_sort_scorer import WrapSortScorer
