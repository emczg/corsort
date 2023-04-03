=======
History
=======

------------
Next release
------------

* Add `Sort.history_comparisons_values_`: history of the pairwise comparisons, in terms of compared values
  (whereas `history_comparisons_` gives the original indices). Similarly, add
  `WrapSortScorer.history_comparisons_values_` and `WrapFullJit.history_comparisons_values_`.
* Add `CorSort.history_leq_`: history of the matrix `leq_` representing the current poset. This is recorded
  if the newly added parameter `record_leq` is True.
* Add `WrapFullJit.history_states_`: history of the state of the list.
* Add `ChainAndY`: poset consisting of a chain and a Y-shape.
* Add `print_corsort_execution`: generate LaTeX code for a CorSort execution.
* `partition` is now stable (in the sense of "stable" sorting), hence also `SortQuick`, `SortAsortQuickselect`,
  and `SortLargestInterval`.

---------------------------------
0.1.0 (2023-02-16): First release
---------------------------------

* Corsort (regular Python or with numba acceleration).
* Classical sorting algorithms: Asort (with quickselect for median selection), Ford-Johnson, quicksort, quicksort with
  priority on the largest interval, merge sort (DFS or BFS).
* Entropy bound.
* Monte-Carlo simulations.
