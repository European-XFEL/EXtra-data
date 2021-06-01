import time
import numpy as np
from extra_data.writer2 import FileWriter, DS


ctrl_grp = 'MID_DET_AGIPD1M-1/x/y'
inst_grp = 'MID_DET_AGIPD1M-1/x/y:output'


class VectorWriterBuffered(FileWriter):
    v = DS(inst_grp, 'azimuthal.profile', (1000,), float)

    class Meta:
        max_train_per_file = 40
        break_into_sequence = True
        class_attrs_interface = True


class VectorWriterDirect(VectorWriterBuffered):
    class Meta:
        buffering = False


def write_vector(cls):
    trains = range(10001, 10101)
    with cls('test_{seq:02d}.h5') as wr:
        # print(wr._meta.max_train_per_file)
        # print(wr.datasets['v'].chunks_autosize(wr._meta.max_train_per_file))
        # add data (funcion kwargs interface)
        for tid in trains:
            # create/compute data for 1000 pulses
            v = np.random.randn(1000, 1000)
            # add data (class attribute interface)
            wr.v = v
            # write train
            wr.write_train(tid, 0)


class ScalarWriterBuffered(FileWriter):
    a = DS(ctrl_grp, 'param.a', (), np.uint64)
    b = DS(ctrl_grp, 'param.b', (), np.uint64)
    c = DS(ctrl_grp, 'param.c', (), np.uint64)
    d = DS(ctrl_grp, 'param.d', (), np.uint64)
    e = DS(ctrl_grp, 'param.e', (), np.uint64)
    f = DS(ctrl_grp, 'param.f', (), np.uint64)
    g = DS(ctrl_grp, 'param.g', (), np.uint64)
    i = DS(ctrl_grp, 'param.i', (), np.uint64)
    k = DS(ctrl_grp, 'param.k', (), np.uint64)
    m = DS(ctrl_grp, 'param.m', (), np.uint64)

    class Meta:
        max_train_per_file = 2000
        break_into_sequence = True
        class_attrs_interface = True


class ScalarWriterDirect(ScalarWriterBuffered):
    class Meta:
        buffering = False


def write_scalars(cls):
    trains = range(10001, 20001)
    with cls('test_{seq:02d}.h5') as wr:
        i = 0
        for tid in trains:
            # add data (class attribute interface)
            wr.add(
                a=i+1, b=i+2, c=i+3, d=i+4, e=i+5,
                f=i+6, g=i+7, i=i+8, k=i+9, m=i+10,
            )
            # write train
            wr.write_train(tid, 0)
            i += 10


def bench(fun, nrep):
    tm = np.zeros(nrep, float)
    for rep in range(nrep):
        t0 = time.monotonic()
        fun()
        t1 = time.monotonic()
        tm[rep] = t1 - t0
    return tm


if __name__ == "__main__":
    print("Medium data: 100 trains by 1000 frames of vectors [float: 1000]")
    tm_buf = bench(lambda: write_vector(VectorWriterBuffered), 5)
    print(f" - buffered: nrep=5, mean={tm_buf.mean():.2f} s, "
          f"std={tm_buf.std():.3g} s")
    tm_dir = bench(lambda: write_vector(VectorWriterDirect), 5)
    print(f" - direct:   nrep=5, mean={tm_dir.mean():.2f} s, "
          f"std={tm_dir.std():.3g} s")

    print("Small data: 10000 trains by 1 frame of 10 scalars [uint64]")
    tm_buf = bench(lambda: write_scalars(ScalarWriterBuffered), 15)
    print(f" - buffered: nrep=15, mean={tm_buf.mean():.2f} s, "
          f"std={tm_buf.std():.3g} s")
    tm_dir = bench(lambda: write_scalars(ScalarWriterDirect), 3)
    print(f" - direct:   nrep=3,  mean={tm_dir.mean():.2f} s, "
          f"std={tm_dir.std():.3g} s")
