
import mmap
import multiprocessing
import multiprocessing.pool
import queue
import threading

import numpy


class MapContext:
    """Context to execute map operations.

    A map operation applies a single callable to each train of the
    targeted DataCollection. The context define the runtime conditions
    for this operation, which may be in a process pool, for example.

    As some of these environments may require special memory semantics,
    the context also provides a method of allocating ndarrays. The only
    abstract method required by an implementation is
    :func:run(kernel, target).
    """

    def array(self, shape, dtype, per_worker=False, **kwargs):
        """High-level allocation.

        Expects a property called n_worker.
        """

        if per_worker:
            shape = (self.n_worker,) + shape

        return self.alloc_array(shape, dtype)

    def alloc_array(self, shape, dtype):
        """Low-level allocation.
        """

        return numpy.zeros(shape, dtype=dtype)

    def map(self, kernel, target, worker_id):
        """Low-level map operation.
        """

        for train_id, data in target.trains():
            kernel(worker_id, train_id, data)

    def run(self, kernel, target):
        raise NotImplementedError('run')


class LocalContext(MapContext):
    """Local map context.

    Runs the map operation directly.
    """
    def __init__(self, n_worker=None):
        self.n_worker = 1

    def run(self, kernel, target):
        self.map(kernel, target, 0)


class MapPoolContext(MapContext):
    """Map context for multiprocessing Pool interface.
    """

    def run(self, kernel, target, id_queue, pool_cls):
        self.kernel = kernel

        for worker_id in range(self.n_worker):
            id_queue.put(worker_id)

        for f in target.files:
            f.close()

        with pool_cls(self.n_worker, self.init_pool) as p:
            p.map(self.map, target.split_trains(self.n_worker))

    def init_pool(self):
        pass


class ThreadContext(MapPoolContext):
    """Map context in a thread pool.
    """

    def __init__(self, n_worker):
        self.n_worker = n_worker

        self.worker_id_map = {}
        self.id_queue = queue.Queue()

    def alloc_array(self, shape, dtype):
        return numpy.zeros(shape, dtype=dtype)

    def run(self, kernel, target):
        super().run(kernel, target, self.id_queue,
                    multiprocessing.pool.ThreadPool)

    def init_pool(self):
        self.worker_id_map[threading.get_ident()] = self.id_queue.get()

    def map(self, target):
        super().map(self.kernel, target,
                    self.worker_id_map[threading.get_ident()])


class ProcessContext(MapPoolContext):
    """Map context in a process pool.
    """

    _instance = None

    def __init__(self, n_worker):
        try:
            self.mp_ctx = multiprocessing.get_context('fork')
        except ValueError:
            raise ValueError('fork context required')

        self.__class__._instance = self

        self.n_worker = n_worker
        self.id_queue = self.mp_ctx.Queue()

    def alloc_array(self, shape, dtype):
        if isinstance(shape, int):
            n_elements = shape
        else:
            n_elements = 1
            for _s in shape:
                n_elements *= _s

        n_bytes = n_elements * numpy.dtype(dtype).itemsize
        n_pages = n_bytes // mmap.PAGESIZE + 1

        buf = mmap.mmap(-1, n_pages * mmap.PAGESIZE,
                        flags=mmap.MAP_SHARED | mmap.MAP_ANONYMOUS,
                        prot=mmap.PROT_READ | mmap.PROT_WRITE)
        return numpy.frombuffer(memoryview(buf)[:n_bytes],
                                dtype=dtype).reshape(shape)

    def run(self, kernel, target):
        super().run(kernel, target, self.id_queue, self.mp_ctx.Pool)

    def init_pool(self):
        self.worker_id = self.id_queue.get()

    @classmethod
    def map(cls, target):
        # map is a classmethod here and fetches its process-local
        # instance, as the instance in the parent process is not
        # actually part of the execution.

        self = cls._instance
        super(cls, self).map(self.kernel, target, self.worker_id)
