import numpy as np
from string import Template
from itertools import permutations
from corsort.cor_sort_delegate import CorSortDelegate
from corsort.wrap_full_jit import WrapFullJit
from corsort.jit_sorts import jit_corsort_drift_max_spaced


def print_legend(k, state, distance):
    if k == 0:
        positioning = ""
    elif k % 2 == 1:
        positioning = Template(r", right = \lrgap of x$ref").safe_substitute(ref=k-1)
    else:
        positioning = Template(r", below = \interspace of x$ref").safe_substitute(ref=k-2)
    state_as_str = "".join([str(element) for element in state])
    print(Template(r"\node[draw$positioning] (x$k) {$X_$k=($state_as_str), \tau_$k=$distance$};").safe_substitute(
        positioning=positioning, k=k, distance=distance, state_as_str=state_as_str))


def transitive_reduction(leq):
    mask_keep = (leq == 1)
    comparisons = [(i, j) for i, j in zip(*np.where(leq == 1)) if i != j]
    for (i, j), (k, l) in permutations(comparisons, 2):
        if j == k:
            mask_keep[i, l] = False
    comparisons = [(i, j) for i, j in zip(*np.where(mask_keep)) if i != j]
    return comparisons


def print_graph(k, leq, history_comparisons, state, perm):
    print(Template(r"\node[above = \intraspace of x$k] (x${k}p) {\execution{").safe_substitute(k=k))
    n_ancestors = (leq > 0).sum(axis=1)
    n_descendants = (leq > 0).sum(axis=0)
    next_comparison = history_comparisons[k] if k < len(history_comparisons) else {}
    rows = []
    for i, x in enumerate(state):
        n_anc = n_ancestors[i]
        n_des = n_descendants[i]
        rows.append(Template("    $val/$borda/$n_des/$n_tot/$select").safe_substitute(
            val=perm[i], borda=n_des - n_anc, n_des=n_des, n_tot=n_anc + n_des,
            select="select" if i in next_comparison else ""
        ))
    print(", \n".join(rows) + "%")
    comparisons = transitive_reduction(leq)
    comparisons_as_str = ", ".join([f"{perm[i]}/{perm[j]}" for i, j in comparisons])
    print(Template(r"}{$comparisons_as_str}};").safe_substitute(comparisons_as_str=comparisons_as_str))


def print_preamble():
    print(r"""
\begin{tikzpicture}
\def\interspace{3cm}
\def\intraspace{.1cm}
\def\lrgap{1.5cm}
\newcommand{\execution}[2]{%
    \begin{tikzpicture}[scale=.7, transform shape]
        \foreach \i/\d/\an/\to/\s [count=\x] in {#1}
        {
            \node[obj] (\i) at (\x, 4*\an/\to) {$\i$};
            \node[above = .0cm of \i, \s] {$\d$};
        }
        \foreach \i/\j in {#2}{\draw[<-] (\i) -- (\j) ;}
    \end{tikzpicture}%
}
    """)


def print_end():
    print(r"\end{tikzpicture}")


def print_corsort_execution(perm):
    corsort = CorSortDelegate(
        sort=WrapFullJit(jit_sort=jit_corsort_drift_max_spaced),
        compute_history=True,
        record_leq=True
    )
    corsort(perm)
    print_preamble()
    for k, (state, distance, leq) in enumerate(zip(corsort.sort.history_states_,
                                                   corsort.history_distances_,
                                                   corsort.history_leq_)):
        print_legend(k, state, distance)
        print_graph(k, leq, corsort.history_comparisons_, state, perm)
    print_end()
