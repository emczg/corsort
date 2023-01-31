import numpy as np


class Sort:
    """
    Abstract class for sorting algorithms.

    Parameters
    ----------
    compute_history: :class:`bool`
        If True, then compute the history of the distance to the sorted array.
    """

    def __init__(self, compute_history=False):
        # Parameters
        self.compute_history = compute_history
        # Computed values
        self.n_ = None
        self.perm_ = None
        self.n_comparisons_ = None
        self.history_distances_ = None

    def __call__(self, perm):
        """
        Sort.

        Parameters
        ----------
        perm: :class:`numpy.ndarray`
            Input permutation to sort. Typically the output of :meth`~numpy.random.permutation`.

        Returns
        -------
        :class:`int`
            Number of comparisons to sort the permutation.
        :class:`list`
            History of the distance between the list and the sorted list.
        """
        if isinstance(perm, list):
            perm = np.array(perm)
        self.n_ = len(perm)
        self.perm_ = perm
        self._call_aux()
        return self.n_comparisons_, self.history_distances_

    def _call_aux(self):
        """
        Must update self.n_comparisons_, self.history_distances_.
        """
        raise NotImplementedError
