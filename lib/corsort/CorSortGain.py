from corsort.CorSort import CorSort


class CorSortGain(CorSort):

    def gain(self, i, j):
        # TODO: in fact this should be only in a sub-class that uses the `gain` method.
        # (For the moment, it is not used in CorSortBorda for example).
        raise NotImplementedError

    def next_compare(self):
        while True:
            gain = self.gain(0, 0)
            arg = None
            for i in range(self.n_):
                for j in range(i + 1, self.n_):
                    ng = self.gain(i, j)
                    if ng > gain:
                        arg = (i, j)
                        gain = ng
            if arg is not None:
                yield arg
            else:
                break
