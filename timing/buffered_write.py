import time
import numpy as np
import h5py
from extra_data.writer2 import FileWriter, DS


class VectorWriterBuffered(FileWriter):
    v = DS('MID_DET_AGIPD1M-1/x/y:output', 'azimuthal.profile', (1000,), float)

    class Meta:
        max_train_per_file = 500
        break_into_sequence = True
        class_attrs_interface = True


class VectorWriterDirect(VectorWriterBuffered):
    class Meta:
        buffering = False


def write_vector_by_trains(cls):
    trains = range(10001, 10101)
    with cls('test_{seq:02d}.h5') as wr:
        # print(wr._meta.max_train_per_file)
        # print(wr.datasets['v'].chunks_autosize(wr._meta.max_train_per_file))
        # add data (funcion kwargs interface)
        for tid in trains:
            # add train
            wr.add_train(tid, 0)
            # create/compute data for 1000 pulses
            v = np.random.randn(1000, 1000)
            # add data (class attribute interface)
            wr.v = v


def write_vector_at_once(cls):
    trains = list(range(10001, 10101))
    ntrain = len(trains)
    with cls('test_{seq:02d}.h5') as wr:
        # print(wr._meta.max_train_per_file)
        # print(wr.datasets['v'].chunks_autosize(wr._meta.max_train_per_file))
        # add data (funcion kwargs interface)
        wr.add_trains(trains, [0] * ntrain)
        v = np.random.randn(ntrain * 1000, 1000)
        wr.add_data([1000]*ntrain, v=v)


class ScalarWriterBuffered(FileWriter):
    a = DS('@ctrl', 'param.a', (), np.uint64)
    b = DS('@ctrl', 'param.b', (), np.uint64)
    c = DS('@ctrl', 'param.c', (), np.uint64)
    d = DS('@ctrl', 'param.d', (), np.uint64)
    e = DS('@ctrl', 'param.e', (), np.uint64)
    f = DS('@ctrl', 'param.f', (), np.uint64)
    g = DS('@ctrl', 'param.g', (), np.uint64)
    i = DS('@ctrl', 'param.i', (), np.uint64)
    k = DS('@ctrl', 'param.k', (), np.uint64)
    m = DS('@ctrl', 'param.m', (), np.uint64)

    class Meta:
        max_train_per_file = 10000
        break_into_sequence = False
        class_attrs_interface = True
        aliases = {
            'ctrl': 'MID_DET_AGIPD1M-1/x/y',
        }


class ScalarWriterDirect(ScalarWriterBuffered):
    class Meta:
        buffering = False


def write_scalars_by_trains(cls):
    trains = range(10001, 20001)
    with cls('test_{seq:02d}.h5') as wr:
        i = 0
        for tid in trains:
            # add train
            wr.add_train(tid, 0)
            # add data (class attribute interface)
            wr.add_train_data(
                a=i+1, b=i+2, c=i+3, d=i+4, e=i+5,
                f=i+6, g=i+7, i=i+8, k=i+9, m=i+10,
            )
            i += 10


def write_scalars_at_once(cls):
    trains = list(range(10001, 20001))
    ntrain = len(trains)
    with cls('test_{seq:02d}.h5') as wr:
        # add train
        wr.add_trains(trains, [0]*ntrain)
        # add data (class attribute interface)
        wr.add_data([1]*ntrain,
            a=np.arange(ntrain)*10 + 1,
            b=np.arange(ntrain)*10 + 2,
            c=np.arange(ntrain)*10 + 3,
            d=np.arange(ntrain)*10 + 4,
            e=np.arange(ntrain)*10 + 5,
            f=np.arange(ntrain)*10 + 6,
            g=np.arange(ntrain)*10 + 7,
            i=np.arange(ntrain)*10 + 8,
            k=np.arange(ntrain)*10 + 9,
            m=np.arange(ntrain)*10 + 10,
        )


def write_with_h5py(ntrain, nds, nitem, dtype, chunks=None):
    with h5py.File('test_plain.h5', 'w') as f:
        for i in range(nds):
            if np.issubdtype(dtype, np.integer):
                a = np.random.randint(0, 127, size=nitem*ntrain, dtype=dtype)
            else:
                a = np.random.rand(nitem*ntrain).astype(dtype)
            ds = f.create_dataset(f'plain/data/{i}', shape=(nitem*ntrain,), chunks=chunks, dtype=dtype)
            ds[:] = a
        f['INDEX/trainId'] = np.zeros(ntrain, dtype=np.uint64)
        f['INDEX/timestamps'] = np.zeros(ntrain, dtype=np.uint64)
        f['INDEX/flags'] = np.zeros(ntrain, dtype=np.uint32)
        count = [nitem] * ntrain
        f['INDEX/group/first'] = np.concatenate([[0], np.cumsum(count)[:-1]])
        f['INDEX/group/count'] = count


def bench(fun, min_rep = 3, max_rep=30, tm_limit = 30):
    tm = []
    rep = 0
    wtime = 0
    while (rep < min_rep or wtime < tm_limit) and rep < max_rep:
        t0 = time.monotonic()
        fun()
        dt = time.monotonic() - t0
        tm.append(dt)
        wtime += dt
        rep += 1
    return np.array(tm)


if __name__ == "__main__":
    chunks = VectorWriterBuffered.datasets['v'].chunks_autosize(VectorWriterBuffered._meta.max_train_per_file)
    print("Medium data: 100 trains by 1000 frames of 1 vector [float: 1000]")
    tm_pln = bench(lambda: write_with_h5py(100, 1, 1000000, float, (np.prod(chunks),)))
    print(f" - h5py:     nrep={len(tm_pln)}, mean={tm_pln.mean():.2f} s, "
          f"std={tm_pln.std():.3g} s")
    tm_buf = bench(lambda: write_vector_by_trains(VectorWriterBuffered))
    print(f" - buffered: nrep={len(tm_buf)}, mean={tm_buf.mean():.2f} s, "
          f"std={tm_buf.std():.3g} s")
    tm_dir = bench(lambda: write_vector_by_trains(VectorWriterDirect))
    print(f" - direct:   nrep={len(tm_dir)}, mean={tm_dir.mean():.2f} s, "
          f"std={tm_dir.std():.3g} s")
    tm_ent = bench(lambda: write_vector_at_once(VectorWriterBuffered))
    print(f" - at once:  nrep={len(tm_ent)}, mean={tm_ent.mean():.2f} s, "
          f"std={tm_ent.std():.3g} s")

    print("Small data: 10000 trains by 1 frame of 10 scalars [uint64]")
    tm_pln = bench(lambda: write_with_h5py(10000, 10, 1, np.uint64))
    print(f" - h5py:     nrep={len(tm_pln)}, mean={tm_pln.mean():.2f} s, "
          f"std={tm_pln.std():.3g} s")
    tm_buf = bench(lambda: write_scalars_by_trains(ScalarWriterBuffered))
    print(f" - buffered: nrep={len(tm_buf)}, mean={tm_buf.mean():.2f} s, "
          f"std={tm_buf.std():.3g} s")
    tm_dir = bench(lambda: write_scalars_by_trains(ScalarWriterDirect))
    print(f" - direct:   nrep={len(tm_dir)},  mean={tm_dir.mean():.2f} s, "
          f"std={tm_dir.std():.3g} s")
    tm_ent = bench(lambda: write_scalars_at_once(ScalarWriterBuffered))
    print(f" - at once:  nrep={len(tm_ent)},  mean={tm_ent.mean():.2f} s, "
          f"std={tm_ent.std():.3g} s")
