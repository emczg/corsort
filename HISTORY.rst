=======
History
=======

-------------------------------------------------------------------
0.1.2 (2023-06-05): Binary Insertion Sort, Multizip Sort, Shellsort
-------------------------------------------------------------------

* `Corsort` and subclasses (i.e. non-jit Corsort algorithms):

  * Add parameter `record_leq`. If True, then record all the states of the `leq_` matrix.
  * Add parameter `final_score`. Scorer used to compute the tentative estimate of the sorted list.

* Add `CorsortChainDecompositionMergeV`: Corsort based on chain decomposition, with "V-shape" merging.
* Add `CorsortChainDecompositionMergeX`: Corsort based on chain decomposition, with "X-shape" merging.
* Add `greedy_chain_decomposition`: greedy chain decomposition.
* Add `longest_chain`: longest chain.
* Add `longest_chain_starting_at`: longest chain starting at a given item.
* Add `multi_merge`: merge consecutive sorted portions of a list, two by two, in alternance. Used for multizip sort.
* Add `scorer_delta` and `scorer_rho`: scorer delta or rho. Mostly used for the `final_score` parameter of `Corsort`.
* Add `SortBinaryInsertion`: binary insertion sort.
* Add `SortMultizip`: multizip sort.
* Add `SortShell`: Shellsort.
* Add `split_pointer_lists`: compute the indices of the boundaries for all the steps of bottom-up (BFS) merge sort.
* Add `transitive_reduction`: transitive reduction of a `leq` matrix.
* `WrapFullJit`: add parameter `record_states`. If True, then record the states of the algorithm.
* Add predefined wrappers using `WrapFullJit`: `JitCorsortBorda`, `JitHeapsort`, `JitCorsortDeltaMaxRho`,
  `JitCorsortDeltaSumRho`, etc.
* Rename `CorSort` to `Corsort`, and similarly for subclasses.
* Rename `print_order` to `print_order_as_letters`.
* Rename `drift` to `delta` and `spaced` to `rho` in all function names in order to match the notations of our papers.
* Rename `scorer_drift` to `jit_scorer_delta` and `scorer_spaced` to `jit_scorer_rho`.
* Rename `plus` to `sum` in jit corsort functions. For example, rename `jit_corsort_drift_plus_spaced` to
  `jit_corsort_delta_sum_rho`.
* Rename `SortMergeBfs` to `SortMergeBottomUp`.
* Rename `SortMergeDfs` to `SortMergeTopDown`.

------------------------------------------
0.1.1 (2023-04-7): More history, ChainAndY
------------------------------------------

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
