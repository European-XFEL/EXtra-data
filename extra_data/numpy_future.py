# compatibility to future numpy features
import numpy as np

from packaging import version

if version.parse(np.version.version) < version.parse("1.20"):

    from numpy.core.overrides import set_module

    def _broadcast_shape(*args):
        """Returns the shape of the arrays that would result from broadcasting the
        supplied arrays against each other.
        """
        # use the old-iterator because np.nditer does not handle size 0 arrays
        # consistently
        b = np.broadcast(*args[:32])
        # unfortunately, it cannot handle 32 or more arguments directly
        for pos in range(32, len(args), 31):
            # ironically, np.broadcast does not properly handle np.broadcast
            # objects (it treats them as scalars)
            # use broadcasting to avoid allocating the full array
            b = np.broadcast_to(0, b.shape)
            b = np.broadcast(b, *args[pos:(pos + 31)])
        return b.shape

    @set_module('numpy')
    def broadcast_shapes(*args):
        """Broadcast the input shapes into a single shape."""
        arrays = [np.empty(x, dtype=[]) for x in args]
        return _broadcast_shape(*arrays)

    def add_future_function_into(numpy_namespace):
        numpy_namespace.broadcast_shapes = broadcast_shapes
else:
    def add_future_function_into(numpy_namespace):
        pass
