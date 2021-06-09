import time
import numpy as np
import h5py
import cProfile
import subprocess

from extra_data.writer2 import FileWriter, DS


class VectorWriterBuffered(FileWriter):
    v = DS('/x/y:output', 'azimuthal.profile', (1000,), float)

    class Meta:
        max_train_per_file = 500
        break_into_sequence = True
        class_attrs_interface = True


class VectorWriterDirect(VectorWriterBuffered):
    class Meta:
        buffering = False


def write_vector_by_trains(cls, trains, data):
    detname = 'MID_DET_AGIPD1M-1'
    with cls('test_{seq:02d}.h5', detname=detname) as wr:
        for i, tid in enumerate(trains):
            # add train
            wr.add_train(tid, 0)
            # add data
            wr.add_train_data(v=data[i])


def write_vector_at_once(cls, trains, data):
    detname = 'MID_DET_AGIPD1M-1'
    ntrain = len(trains)
    nitem = data.shape[1]
    nrec = ntrain * nitem
    shape = (nrec,) + data.shape[2:]
    with cls('test_{seq:02d}.h5', detname=detname) as wr:
        # add data
        wr.add_trains(trains, [0] * ntrain)
        wr.add_data([nitem]*ntrain, v=data.reshape(*shape))


def write_vector_h5py(trains, data, chunks):
    ntrain = len(trains)
    nitem = data.shape[1]
    nrec = ntrain * nitem
    count = np.array([nitem] * ntrain, dtype=np.uint64)
    first = np.roll(np.cumsum(count), 1)
    first[0] = 0
    shape = (nrec,) + data.shape[2:]
    with h5py.File('test_plain.h5', 'w') as f:
        f['INDEX/trainId'] = np.array(trains, dtype=np.uint64)
        f['INDEX/timestamps'] = np.zeros(ntrain, dtype=np.uint64)
        f['INDEX/flags'] = np.zeros(ntrain, dtype=np.uint32)
        f['INDEX/group/first'] = first
        f['INDEX/group/count'] = count

        ds = f.create_dataset('plain/vector', shape=shape,
                              chunks=chunks, dtype=data.dtype)
        ds.write_direct(data.reshape(*shape), np.s_[:], np.s_[:])


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


def write_scalars_by_trains(cls, trains, data):
    detname = 'MID_DET_AGIPD1M-1'
    with cls('test_{seq:02d}.h5', detname=detname) as wr:
        for i, tid in enumerate(trains):
            # add train
            wr.add_train(tid, 0)
            # add data (class attribute interface)
            wr.add_train_data(
                a=data.a[i], b=data.b[i], c=data.c[i], d=data.d[i],
                e=data.e[i], f=data.f[i], g=data.g[i], i=data.i[i],
                k=data.k[i], m=data.m[i],
            )


def write_scalars_at_once(cls, trains, data):
    detname = 'MID_DET_AGIPD1M-1'
    ntrain = len(trains)
    with cls('test_{seq:02d}.h5', detname=detname) as wr:
        # add train
        wr.add_trains(trains, [0]*ntrain)
        # add data (class attribute interface)
        wr.add_data(
            [1]*ntrain,
            a=data.a, b=data.b, c=data.c, d=data.d, e=data.e,
            f=data.f, g=data.g, i=data.i, k=data.k, m=data.m,
        )


def write_scalars_h5py(trains, data, chunks):
    ntrain = len(trains)
    count = np.array([1] * ntrain, dtype=np.uint64)
    first = np.arange(ntrain, dtype=np.uint64)
    with h5py.File('test_plain.h5', 'w') as f:
        f['INDEX/trainId'] = np.array(trains, dtype=np.uint64)
        f['INDEX/timestamps'] = np.zeros(ntrain, dtype=np.uint64)
        f['INDEX/flags'] = np.zeros(ntrain, dtype=np.uint32)
        f['INDEX/group/first'] = first
        f['INDEX/group/count'] = count

        ds = {}
        for n in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'i', 'k', 'm']:
            v = getattr(data, n)
            ds[n] = f.create_dataset(f'plain/scalar/{n}', shape=v.shape,
                                     dtype=v.dtype, chunks=chunks)
        for n in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'i', 'k', 'm']:
            ds[n].write_direct(getattr(data, n), np.s_[:], np.s_[:])


def cumstat(stats, pfile):
    tm, tm_wr2, tm_wr2_h5, tm_h5 = 0, 0, 0, 0
    for fun, stat in stats.items():
        if pfile in fun[0] and not stat[4]:
            tm += stat[3]
        if 'extra_data/writer2.py' in fun[0]:
            for prn, pstat in stat[4].items():
                if pfile in prn[0]:
                    tm_wr2 += pstat[3]
        if 'h5py' in fun[0]:
            for prn, pstat in stat[4].items():
                if 'extra_data/writer2.py' in prn[0]:
                    tm_wr2_h5 += pstat[3]
                if pfile in prn[0]:
                    tm_h5 += pstat[3]
    tm -= tm_wr2 + tm_h5
    return tm, tm_wr2 - tm_wr2_h5, tm_wr2_h5, tm_h5


def bench(fun, tag, min_rep=3, max_rep=30, tm_limit=30):
    pr = cProfile.Profile()
    tm = []
    rep = 0
    wtime = 0
    while (rep < min_rep or wtime < tm_limit) and rep < max_rep:
        t0 = time.monotonic()
        pr.runcall(fun)
        dt = time.monotonic() - t0
        tm.append(dt)
        wtime += dt
        rep += 1

    pr.create_stats()

    tm_bnc, tm_wr2, tm_wr2_h5, tm_h5 = cumstat(pr.stats, __file__)

    pr.dump_stats(f"wr2_{tag}.pstats")
    subprocess.run(f"gprof2dot -f pstats wr2_{tag}.pstats | "
                   f"dot -Tpng -o wr2_{tag}.png", shell=True)

    return np.array(tm), tm_bnc, tm_wr2, tm_wr2_h5, tm_h5


if __name__ == "__main__":
    max_trains = VectorWriterBuffered._meta.max_train_per_file
    vchunks = DS._chunks_autosize(max_trains, (1000,), np.dtype(float), 1)

    vtrains = list(range(10001, 10101))
    nvtrain = len(vtrains)
    vec = np.random.rand(nvtrain, 1000, 1000)

    max_trains = ScalarWriterBuffered._meta.max_train_per_file
    schunks = DS._chunks_autosize(max_trains, (), np.dtype(np.uint64), 0)

    strains = list(range(10001, 20001))
    nstrain = len(strains)
    scl = type('ScalarData', (), {
        'a': np.random.randint(0, 16000, size=nstrain, dtype=np.uint64),
        'b': np.random.randint(0, 16000, size=nstrain, dtype=np.uint64),
        'c': np.random.randint(0, 16000, size=nstrain, dtype=np.uint64),
        'd': np.random.randint(0, 16000, size=nstrain, dtype=np.uint64),
        'e': np.random.randint(0, 16000, size=nstrain, dtype=np.uint64),
        'f': np.random.randint(0, 16000, size=nstrain, dtype=np.uint64),
        'g': np.random.randint(0, 16000, size=nstrain, dtype=np.uint64),
        'i': np.random.randint(0, 16000, size=nstrain, dtype=np.uint64),
        'k': np.random.randint(0, 16000, size=nstrain, dtype=np.uint64),
        'm': np.random.randint(0, 16000, size=nstrain, dtype=np.uint64),
    })

    tests = {
        'scalar': ("small data: 10000 trains by 1 frame of 10 scalars [uint64]", {
            'hdflib': lambda: write_scalars_h5py(strains, scl, schunks),
            'buffer': lambda: write_scalars_by_trains(ScalarWriterBuffered, strains, scl),
            'direct': lambda: write_scalars_by_trains(ScalarWriterDirect, strains, scl),
            'entire': lambda: write_scalars_at_once(ScalarWriterBuffered, strains, scl),
        }),
        'vector': ("medium data: 100 trains by 1000 frames of 1 vector [float: 1000]", {
            'hdflib': lambda: write_vector_h5py(vtrains, vec, vchunks),
            'buffer': lambda: write_vector_by_trains(VectorWriterBuffered, vtrains, vec),
            'direct': lambda: write_vector_by_trains(VectorWriterDirect, vtrains, vec),
            'entire': lambda: write_vector_at_once(VectorWriterBuffered, vtrains, vec),
        }),
    }

    notes = {}
    print("Data(*)   |Variant   |Nrep  |mean(T), s  |std(T), s   |"
          "T_bench, s  |T_wr2, s    |T_wr2_h5, s |T_pure_h5,s ")
    print("----------+----------+------+------------+------------+"
          "------------+------------+------------+------------")
    for dsize, (note, subtests) in tests.items():
        notes[dsize] = note
        for ttype, fun in subtests.items():
            tag = dsize[:3] + '_' + ttype[:3]
            tm, tm_bnc, tm_wr2, tm_wr2_h5, tm_h5 = bench(fun, tag)
            nrep, mean_tm, std_tm = len(tm), tm.mean(), tm.std()
            print(f"{dsize:<10s}|{ttype:<10s}|{nrep:>6d}|{mean_tm:>12.2f}|"
                  f"{std_tm:>12.3g}|{tm_bnc/nrep:12.3g}|{tm_wr2/nrep:12.3g}|"
                  f"{tm_wr2_h5/nrep:12.3g}|{tm_h5/nrep:12.3g}")

    print()
    for dsize, note in notes.items():
        print(f"(*) {dsize} - {note}")
