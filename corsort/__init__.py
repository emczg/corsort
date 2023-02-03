"""Top-level package for Corsort."""

__author__ = """Emma Caizergues"""
__email__ = 'emma.caizergues@gmail.com'
__version__ = '0.1.0'


from corsort.cor_sort import CorSort
from corsort.cor_sort_borda import CorSortBorda
from corsort.cor_sort_delegate import CorSortDelegate
from corsort.cor_sort_gain import CorSortGain
from corsort.cor_sort_gain_lexi import CorSortGainLexi
from corsort.distance_to_sorted_array import distance_to_sorted_array
from corsort.entropy_bound import entropy_bound
from corsort.jit_sorts import jit_corsort_borda
from corsort.montecarlo import print_res, evaluate, evaluate_convergence, evaluate_comparisons
from corsort.scorer_spaced import scorer_spaced
from corsort.sort import Sort
from corsort.sort_ford_johnson import SortFordJohnson
from corsort.sort_quick import SortQuick
from corsort.wrap_full_jit import WrapFullJit
from corsort.wrap_sort_scorer import WrapSortScorer

from corsort.sub_package_1.my_class_1 import MyClass1
from corsort.sub_package_2.my_class_2 import MyClass2
from corsort.sub_package_2.my_class_3 import MyClass3
