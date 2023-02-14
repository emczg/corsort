from corsort.cor_sort import CorSort


class CorSortGain(CorSort):
    """
    CorSort based on a `gain` function.
    """

    def gain(self, i, j):
        """
        Gain to be expected when comparing perm[i] and perm[j].

        Notes
        -----
        The gain function must be minimal for all pairs (i, j) whose comparison is already known,
        including pairs of the form (i, i).

        Parameters
        ----------
        i: :class:`int`
            Index of the first item.
        j: :class:`int`
            Index of the second item.

        Returns
        -------
        comparable
            A comparable object, e.g. a number.
        """
        raise NotImplementedError

    def next_compare(self):
        while True:
            # Find pair (i, j) with maximal gain.
            max_gain, argmax_i, argmax_j = max(
                [(self.gain(i, j), i, j) for i in range(self.n_) for j in range(i + 1, self.n_)],
                key=lambda x: x[0]
            )
            if max_gain > self.gain(0, 0):
                yield argmax_i, argmax_j
            else:  # pragma: no cover
                break
