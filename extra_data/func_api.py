
import mmap
import multiprocessing
import multiprocessing.pool
import queue
import threading

import numpy


try:
    mp_ctx = multiprocessing.get_context('fork')
except ValueError:
    DEFAULT_PL_METHOD = 'threading'
else:
    DEFAULT_PL_METHOD = 'processes'

#DEFAULT_PL_WORKER = multiprocessing.cpu_count() // 3
DEFAULT_PL_WORKER = 5



def parallelized(func):
    def parallelized_func(kernel, target, **kwargs):
        pl_worker = min(get_pl_worker(kwargs), len(target.train_ids))
        pl_method = get_pl_method(kwargs)

        # Don't bother with a pool for one worker.
        if pl_worker == 1:
            func(target, kernel, 0)
            return

        if pl_method == 'processes':
            queue_class = mp_ctx.Queue
            pool_class = mp_ctx.Pool

            # - Must be a top-level function to be picklable.
            # - Global variable shared across fork() allows any
            #   kind of kernel, even lambda's.

            def init_func():
                global _func, _kernel, _worker_id
                _func = func
                _kernel = kernel
                _worker_id = id_queue.get()

            run_func = _process_run

        elif pl_method == 'threads':
            queue_class = queue.Queue
            pool_class = multiprocessing.pool.ThreadPool
            worker_id_map = {}

            def init_func():
                worker_id_map[threading.get_ident()] = id_queue.get()

            def run_func(target):
                func(target, kernel, worker_id_map[threading.get_ident()])

        id_queue = queue_class()
        for worker_id in range(pl_worker):
            id_queue.put(worker_id)

        # IMPORTANT!
        # Force each worker to open its own file handles, as h5py horribly
        # breaks if the same handle is used across threads or even processes.
        for f in target.files:
            f.close()

        with pool_class(pl_worker, init_func) as p:
            p.map(run_func, target.split_trains(pl_worker))

    return parallelized_func


def _process_run(target):
    _func(target, _kernel, _worker_id)

@parallelized
def map_kernel_by_train(target, kernel, worker_id):
    for train_id, data in target.trains():
        kernel(worker_id, train_id, data)


def get_pl_worker(kwargs, default=None):
    try:
        pl_worker = int(kwargs['pl_worker'])
    except KeyError:
        if default is not None:
            pl_worker = default
        else:
            pl_worker = DEFAULT_PL_WORKER

    except ValueError:
        raise ValueError('invalid pl_worker value') from None

    else:
        del kwargs['pl_worker']

    return pl_worker


def get_pl_method(kwargs, default=None):
    try:
        pl_method = kwargs['pl_method']
    except KeyError:
        if default is not None:
            pl_method = default
        else:
            pl_method = DEFAULT_PL_METHOD

    else:
        del kwargs['pl_method']

    if pl_method not in ('processes', 'threads'):
        raise ValueError('invalid parallelization method') from None

    return pl_method


def get_pl_env(kwargs, pl_worker=None, pl_method=None):
    return get_pl_worker(kwargs, pl_worker), get_pl_method(kwargs, pl_method)


def alloc_array(shape, dtype, per_worker=False, **kwargs):
    pl_method = get_pl_method(kwargs)

    if per_worker:
        pl_worker = get_pl_worker(kwargs)
        shape = (pl_worker,) + shape

    if pl_method == 'processes':
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

    elif pl_method == 'threads':
        return numpy.zeros(shape, dtype=dtype)
