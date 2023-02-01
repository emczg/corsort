"""Top-level package for Corsort."""

__author__ = """Emma Caizergues"""
__email__ = 'emma.caizergues@gmail.com'
__version__ = '0.1.0'


from corsort.JitSortBorda import JitSortBorda
from corsort.entropy_bound import entropy_bound
from corsort.SortQuick import _quicksort, SortQuick
from corsort.montecarlo import evaluate_convergence, evaluate_comparisons
from corsort.distance_to_sorted_array import distance_to_sorted_array
from corsort.sub_package_1.my_class_1 import MyClass1
from corsort.sub_package_2.my_class_2 import MyClass2
from corsort.sub_package_2.my_class_3 import MyClass3