import h5py
import os.path as osp
import numpy as np
from tempfile import TemporaryDirectory
from testpath import assert_isfile

from extra_data import RunDirectory, H5File
from extra_data.writer2 import FileWriter, DS


ctrl_grp = 'MID_DET_AGIPD1M-1/x/y'
inst_grp = 'MID_DET_AGIPD1M-1/x/y:output'
nbin = 1000


class MyFileWriter(FileWriter):
    gv = DS(ctrl_grp, 'geom.fragmentVectors', (10,100), float)
    nb = DS(ctrl_grp, 'param.numberOfBins', (), np.uint64)
    rlim = DS(ctrl_grp, 'param.radiusRange', (2,), float)

    tid = DS(inst_grp, 'azimuthal.trainId', (), np.uint64)
    pid = DS(inst_grp, 'azimuthal.pulseId', (), np.uint64)
    v = DS(inst_grp, 'azimuthal.profile', (nbin,), float)

    class Meta:
        max_train_per_file = 10
        break_into_sequence = True
        #warn_on_missing_data = True


def test_writer2():

    with TemporaryDirectory() as td:
        new_file = osp.join(td, 'test{seq:03d}.h5')

        trains = list(range(100001,100026))
        pulses = list(range(4, 16, 4))
        ntrain = len(trains)
        npulse = len(pulses)

        gv = np.random.randn(10, 100)
        vref = []

        with MyFileWriter(new_file) as wr:
            # add data:
            # 1. class attribute interface
            # wr.gv = gv
            # wr.nb = nbin
            # wr.rlim = (0.003, 0.016)
            # 2. funcion kwargs interface

            for tid in trains:
                # create/compute data
                v = np.random.randn(npulse, nbin)
                vref.append(v)
                # add train
                wr.add_train(tid, 0)
                # add data
                wr.add_train_data(gv=gv, nb=nbin, rlim=(0.003, 0.016))
                
                wr.tid = [tid] * npulse
                wr.pid = pulses
                wr.v = v

        vref = np.concatenate(vref, 0)

        if MyFileWriter._meta.break_into_sequence:
            tpf = MyFileWriter._meta.max_train_per_file
            nseq = ntrain // tpf + int((ntrain % tpf) > 0)
        else:
            nseq, tpf = 1, ntrain
            
        for seq in range(nseq):
            assert_isfile(new_file.format(seq=seq))

        with RunDirectory(td) as run:
            np.testing.assert_array_equal(run.train_ids,
                np.array(trains, dtype=np.uint64))

            assert ctrl_grp in run.control_sources
            assert inst_grp in run.instrument_sources
            
            assert len(run.files) == nseq
            for i in range(nseq):
                seq = int(run.files[i].filename[-6:-3])
                t0, tN = seq * tpf, min(seq * tpf + tpf, ntrain)
                np.testing.assert_array_equal(
                    run.files[i].train_ids, 
                    np.array(trains[t0:tN], dtype=np.uint64))

            nb = run.get_array(ctrl_grp, 'param.numberOfBins')
            np.testing.assert_array_equal(
                nb.values, np.full([ntrain], nbin, dtype=np.uint64))

            rlim = run.get_array(ctrl_grp, 'param.radiusRange')
            np.testing.assert_array_equal(rlim.values, np.broadcast_to(
                np.array([0.003, 0.016], dtype=float), (ntrain,2)))

            frg = run.get_array(ctrl_grp, 'geom.fragmentVectors')
            np.testing.assert_array_equal(frg.values, np.broadcast_to(
                gv, (ntrain,) + gv.shape))

            tt, pp = np.meshgrid(trains, pulses, indexing='ij')

            tid = run.get_array(inst_grp, 'azimuthal.trainId')
            np.testing.assert_array_equal(tid.values, tt.ravel())

            pid = run.get_array(inst_grp, 'azimuthal.pulseId')
            np.testing.assert_array_equal(pid.values, pp.ravel())

            v = run.get_array(inst_grp, 'azimuthal.profile')
            np.testing.assert_array_equal(v.values, vref)
