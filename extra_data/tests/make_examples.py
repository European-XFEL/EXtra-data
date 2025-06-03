import os
import os.path as osp

import h5py
import numpy as np

from .mockdata import write_file
from .mockdata.adc import ADC
from .mockdata.agipd import AGIPD1MFPGA, AGIPD1MPSC, AGIPD500KFPGA, AGIPDMDL
from .mockdata.base import write_base_index
from .mockdata.basler_camera import BaslerCamera as BaslerCam
from .mockdata.dctrl import DCtrl
from .mockdata.detectors import AGIPDModule, DSSCModule, LPDModule
from .mockdata.gauge import Gauge
from .mockdata.gec_camera import GECCamera
from .mockdata.imgfel import IMGFELCamera, IMGFELMotor
from .mockdata.jungfrau import (
    JUNGFRAUControl, JUNGFRAUModule, JUNGFRAUMonitor, JUNGFRAUPower
)
from .mockdata.motor import Motor
from .mockdata.mpod import MPOD
from .mockdata.proc import ReconstructedDLD6
from .mockdata.tsens import TemperatureSensor
from .mockdata.uvlamp import UVLamp
from .mockdata.xgm import XGM

vlen_bytes = h5py.special_dtype(vlen=bytes)


def make_metadata(h5file, data_sources, chunksize=16):
    N = len(data_sources)
    if N % chunksize:
        N += chunksize - (N % chunksize)

    root = [ds.split('/', 1)[0] for ds in data_sources]
    devices = [ds.split('/', 1)[1] for ds in data_sources]

    sources_ds = h5file.create_dataset('METADATA/dataSourceId', (N,),
                                       dtype=vlen_bytes, maxshape=(None,))
    sources_ds[:len(data_sources)] = data_sources
    root_ds = h5file.create_dataset('METADATA/root', (N,),
                                    dtype=vlen_bytes, maxshape=(None,))
    root_ds[:len(data_sources)] = root
    devices_ds = h5file.create_dataset('METADATA/deviceId', (N,),
                                       dtype=vlen_bytes, maxshape=(None,))
    devices_ds[:len(data_sources)] = devices


def make_agipd_example_file(path, format_version='0.5'):
    """Make the structure of a data file from the AGIPD detector

    Based on /gpfs/exfel/d/proc/XMPL/201750/p700000/r0803/CORR-R0803-AGIPD07-S00000.h5

    This has the old index format (first/last/status), whereas the other examples
    have the newer (first/count) format.
    """
    f = h5py.File(path, 'w')

    slow_channels = ['header', 'detector', 'trailer']
    channels = slow_channels + ['image']
    train_ids = np.arange(10000, 10250)   # Real train IDs are ~10^9

    # RUN - empty in the example I'm working from
    f.create_group('RUN')

    # METADATA - lists the data sources in this file
    make_metadata(f, ['INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/'+ch
                      for ch in channels])

    def make_train_ids(path):
        d = f.create_dataset(path, (256,), 'u8', maxshape=(None,))
        d[:250] = train_ids

    # INDEX - matching up data to train IDs
    write_base_index(f, 250, format_version=format_version)
    for ch in channels:
        grp_name = 'INDEX/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/%s/' % ch
        first = f.create_dataset(grp_name + 'first', (256,), 'u8', maxshape=(None,))
        last = f.create_dataset(grp_name + 'last', (256,), 'u8', maxshape=(None,))
        status = f.create_dataset(grp_name + 'status', (256,), 'u4', maxshape=(None,))
        if ch in slow_channels:
            first[:250] = np.arange(250)
            last[:250] = np.arange(250)
        else:
            first[:250] = np.arange(0, 16000, 64)
            last[:250] = np.arange(63, 16000, 64)
        status[:250] = 1

    # INSTRUMENT - the data itself
    #   first, train IDs for each channel
    for ch in slow_channels:
        make_train_ids('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/%s/trainId' % ch)
    fast_tids = f.create_dataset('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/image/trainId',
                                    (16000, 1), 'u8')
    fast_tids[:,0] = np.repeat(train_ids, 64)

    # TODO: Not sure what this is, but it has quite a regular structure.
    # 5408 = 13 x 13 x 32
    f.create_dataset('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/detector/data',
                        (256, 5408), 'u1', maxshape=(None, 5408))

    f.create_dataset('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/header/dataId',
                        (256,), 'u8', maxshape=(None,))  # Empty in example
    linkId = f.create_dataset('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/header/linkId',
                        (256,), 'u8', maxshape=(None,))
    linkId[:250] = 18446744069414584335  # Copied from example
    f.create_dataset('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/header/magicNumberBegin',
                        (256, 8), 'i1', maxshape=(None, 8))  # TODO: fill in data
    vmaj = f.create_dataset('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/header/majorTrainFormatVersion',
                        (256,), 'u4', maxshape=(None,))
    vmaj[:250] = 1
    vmin = f.create_dataset('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/header/minorTrainFormatVersion',
                        (256,), 'u4', maxshape=(None,))
    vmin[:250] = 0
    pc = f.create_dataset('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/header/pulseCount',
                        (256,), 'u8', maxshape=(None,))
    pc[:250] = 64
    f.create_dataset('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/header/reserved',
                        (256, 16), 'u1', maxshape=(None, 16))  # Empty in example

    cellId = f.create_dataset('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/image/cellId',
                        (16000, 1), 'u2')
    cellId[:, 0] = np.tile(np.arange(64), 250)
    # The data itself
    f.create_dataset('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/image/data',
                        (16000, 512, 128), 'f4')
    f.create_dataset('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/image/gain',
                        (16000, 512, 128), 'u1')
    length = f.create_dataset('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/image/length',
                        (16000, 1), 'u4', maxshape=(None, 1))
    length[:] = 262144  # = 512*128*4(bytes) ?
    f.create_dataset('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/image/mask',
                        (16000, 512, 128, 3), 'u1')  # TODO: values 128 or 0
    pulseId = f.create_dataset('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/image/pulseId',
                        (16000, 1), 'u8')
    # In the real data, these are unevenly spaced, but this is close enough
    pulseId[:, 0] = np.tile(np.linspace(0, 125, 64, dtype='u8'), 250)
    f.create_dataset('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/image/status',
                        (16000, 1), 'u2')  # Empty in example

    f.create_dataset('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/trailer/checksum',
                        (256, 16), 'i1', maxshape=(None, 16))  # Empty in example
    f.create_dataset('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/trailer/magicNumberEnd',
                        (256, 8), 'i1', maxshape=(None, 8))  # TODO: fill in data
    f.create_dataset('INSTRUMENT/SPB_DET_AGIPD1M-1/DET/7CH0:xtdf/trailer/status',
                        (256,), 'u8', maxshape=(None,))  # Empty in example

def make_fxe_da_file(path, format_version='0.5', firsttrain=10000):
    """Make the structure of a file with non-detector data from the FXE experiment

    Based on .../FXE/201830/p900023/r0450/RAW-R0450-DA01-S00001.h5
    """

    write_file(path, [
        XGM('SA1_XTD2_XGM/DOOCS/MAIN'),
        XGM('SPB_XTD9_XGM/DOOCS/MAIN'),
        GECCamera('FXE_XAD_GEC/CAM/CAMERA'),
        GECCamera('FXE_XAD_GEC/CAM/CAMERA_NODATA', nsamples=0)
    ], ntrains=400, chunksize=200, firsttrain=firsttrain, format_version=format_version)

def make_sa3_da_file(path, ntrains=500, format_version='0.5'):
    """Make the structure of a file with non-detector data from SASE3 tunnel

    Based on .../SA3/201830/p900026/r0317/RAW-R0317-DA01-S00000.h5
    """
    write_file(path, [
        ADC('SA3_XTD10_MCP/ADC/1', nsamples=0, channels=(
            'channel_3.output/data',
            'channel_5.output/data',
            'channel_9.output/data',
        )),
        UVLamp('SA3_XTD10_MCP/DCTRL/UVLAMP'),
        Motor('SA3_XTD10_MCP/MOTOR/X2'),
        TemperatureSensor('SA3_XTD10_VAC/TSENS/S30100K'),
        TemperatureSensor('SA3_XTD10_VAC/TSENS/S30160K'),
        TemperatureSensor('SA3_XTD10_VAC/TSENS/S30180K'),
        TemperatureSensor('SA3_XTD10_VAC/TSENS/S30190K'),
        TemperatureSensor('SA3_XTD10_VAC/TSENS/S30200K'),
        TemperatureSensor('SA3_XTD10_VAC/TSENS/S30250K'),
        TemperatureSensor('SA3_XTD10_VAC/TSENS/S30260K'),
        TemperatureSensor('SA3_XTD10_VAC/TSENS/S30280K'),
        TemperatureSensor('SA3_XTD10_VAC/TSENS/S30300K'),
        Gauge('SA3_XTD10_VAC/GAUGE/G30470D_IN'),
        Gauge('SA3_XTD10_VAC/GAUGE/G30480D_IN'),
        Gauge('SA3_XTD10_VAC/GAUGE/G30490D_IN'),
        Gauge('SA3_XTD10_VAC/GAUGE/G30500P'),
        Gauge('SA3_XTD10_VAC/GAUGE/G30510C'),
        DCtrl('SA3_XTD10_VAC/DCTRL/D6_APERT_IN_OK'),
        DCtrl('SA3_XTD10_VAC/DCTRL/D12_APERT_IN_OK'),
        XGM('SA3_XTD10_XGM/XGM/DOOCS'),
        IMGFELCamera('SA3_XTD10_IMGFEL/CAM/BEAMVIEW', nsamples=0),
        IMGFELCamera('SA3_XTD10_IMGFEL/CAM/BEAMVIEW2', nsamples=250),
        IMGFELCamera('SA3_XTD10_IMGFEL/CAM/BEAMVIEW3', nsamples=200),
        IMGFELMotor('SA3_XTD10_IMGFEL/MOTOR/FILTER'),
        IMGFELMotor('SA3_XTD10_IMGFEL/MOTOR/SCREEN'),
        MPOD('SA3_XTD10_MCP/MCPS/MPOD'),
        ReconstructedDLD6('SA3_REMI_DLD6/DET/TOP'),  # RUN-only source.
    ], ntrains=ntrains, chunksize=50, format_version=format_version)

def make_da_file_with_empty_source(path, ntrains=500, format_version='0.5'):
    write_file(path, [
        ADC('SA3_XTD10_MCP/ADC/1', nsamples=0, channels=(
            'channel_3.output/data',
            'channel_5.output/data',
            'channel_9.output/data',
        )),
        UVLamp('SA3_XTD10_MCP/DCTRL/UVLAMP'),
        Motor('SA3_XTD10_MCP/MOTOR/X2'),
        TemperatureSensor('SA3_XTD10_VAC/TSENS/S30100K'),
        Gauge('SA3_XTD10_VAC/GAUGE/G30510C'),
        Gauge('SA3_XTD10_VAC/GAUGE/G30520C', no_ctrl_data=True),
        DCtrl('SA3_XTD10_VAC/DCTRL/D6_APERT_IN_OK'),
        XGM('SA3_XTD10_XGM/XGM/DOOCS'),
        IMGFELCamera('SA3_XTD10_IMGFEL/CAM/BEAMVIEW', nsamples=0),
        IMGFELCamera('SA3_XTD10_IMGFEL/CAM/BEAMVIEW2', nsamples=250),
        IMGFELMotor('SA3_XTD10_IMGFEL/MOTOR/FILTER'),
        MPOD('SA3_XTD10_MCP/MCPS/MPOD'),
    ], ntrains=ntrains, chunksize=50, format_version=format_version)

def make_data_file_bad_device_name(path, format_version='0.5'):
    """Not all devices have the Karabo standard A/B/C naming convention"""
    write_file(path, [
        BaslerCam('SPB_IRU_SIDEMIC_CAM', sensor_size=(1000, 1000))
    ], ntrains=500, chunksize=50, format_version=format_version)

def make_agipd_file(path, format_version='0.5'):
    write_file(path, [
        AGIPDModule('SPB_DET_AGIPD1M-1/DET/0CH0', frames_per_train=64)
    ], ntrains=486, chunksize=32, format_version=format_version)

def make_lpd_file(path, format_version='0.5'):
    write_file(path, [
        LPDModule('FXE_DET_LPD1M-1/DET/0CH0', frames_per_train=128)
    ], ntrains=480, chunksize=32, format_version=format_version)

def make_fxe_run(dir_path, raw=True, format_version='0.5'):
    prefix = 'RAW' if raw else 'CORR'
    for modno in range(16):
        path = osp.join(dir_path,
                        '{}-R0450-LPD{:0>2}-S00000.h5'.format(prefix, modno))
        write_file(path, [
            LPDModule('FXE_DET_LPD1M-1/DET/{}CH0'.format(modno), raw=raw,
                      frames_per_train=128)
        ], ntrains=480, chunksize=32, format_version=format_version)

    if not raw:
        return
    write_file(osp.join(dir_path, 'RAW-R0450-DA01-S00000.h5'), [
        XGM('SA1_XTD2_XGM/DOOCS/MAIN'),
        XGM('SPB_XTD9_XGM/DOOCS/MAIN'),
        GECCamera('FXE_XAD_GEC/CAM/CAMERA'),
        GECCamera('FXE_XAD_GEC/CAM/CAMERA_NODATA', nsamples=0),
    ], ntrains=400, chunksize=200, format_version=format_version)
    write_file(osp.join(dir_path, '{}-R0450-DA01-S00001.h5'.format(prefix)), [
        XGM('SA1_XTD2_XGM/DOOCS/MAIN'),
        XGM('SPB_XTD9_XGM/DOOCS/MAIN'),
        GECCamera('FXE_XAD_GEC/CAM/CAMERA'),
        GECCamera('FXE_XAD_GEC/CAM/CAMERA_NODATA', nsamples=0),
    ], ntrains=80, firsttrain=10400, chunksize=200, format_version=format_version)

def make_lpd_parallelgain_run(dir_path, raw=True, format_version='0.5'):
    prefix = 'RAW' if raw else 'CORR'
    for modno in range(16):
        path = osp.join(dir_path,
                        '{}-R0450-LPD{:0>2}-S00000.h5'.format(prefix,
                                                              modno))
        write_file(path, [
            LPDModule('FXE_DET_LPD1M-1/DET/{}CH0'.format(modno), raw=raw,
                      frames_per_train=300)
        ], ntrains=100, chunksize=32, format_version=format_version)


def make_lpd_run_mini_missed_train(dir_path):
    write_file(osp.join(dir_path, 'RAW-R0450-LPD00-S00000.h5'), [
        LPDModule('FXE_DET_LPD1M-1/DET/0CH0', frames_per_train=10),
    ], ntrains=5, chunksize=5, format_version='1.0')
    mod1_f = osp.join(dir_path, 'RAW-R0450-LPD01-S00000.h5')
    write_file(mod1_f, [
        LPDModule('FXE_DET_LPD1M-1/DET/1CH0', frames_per_train=10),
    ], ntrains=4, chunksize=5, format_version='1.0')

    # Modify the file for module 1, as if it missed train 10002
    # & fill some data to check in the test.
    with h5py.File(mod1_f, 'r+') as f:
        f['INDEX/trainId'][:4] = [10000, 10001, 10003, 10004]
        mod1_dset = f['INSTRUMENT/FXE_DET_LPD1M-1/DET/1CH0:xtdf/image/data']
        mod1_dset[8::10, 0, 0, 0] = np.arange(1, 5)


def make_spb_run(dir_path, raw=True, sensor_size=(1024, 768), format_version='0.5'):
    prefix = 'RAW' if raw else 'CORR'
    for modno in range(16):
        path = osp.join(dir_path,
                        '{}-R0238-AGIPD{:0>2}-S00000.h5'.format(prefix, modno))
        write_file(path, [
            AGIPDModule('SPB_DET_AGIPD1M-1/DET/{}CH0'.format(modno), raw=raw,
                         frames_per_train=64)
            ], ntrains=64, chunksize=32, format_version=format_version)
    if not raw:
        return
    write_file(osp.join(dir_path, '{}-R0238-DA01-S00000.h5'.format(prefix)),
               [ XGM('SA1_XTD2_XGM/DOOCS/MAIN'),
                 XGM('SPB_XTD9_XGM/DOOCS/MAIN'),
                 BaslerCam('SPB_IRU_CAM/CAM/SIDEMIC', sensor_size=sensor_size)
               ], ntrains=32, chunksize=32, format_version=format_version)

    write_file(osp.join(dir_path, '{}-R0238-DA01-S00001.h5'.format(prefix)),
               [ XGM('SA1_XTD2_XGM/DOOCS/MAIN'),
                 XGM('SPB_XTD9_XGM/DOOCS/MAIN'),
                 BaslerCam('SPB_IRU_CAM/CAM/SIDEMIC', sensor_size=sensor_size)
               ], ntrains=32, firsttrain=10032, chunksize=32,
               format_version=format_version)

def make_reduced_spb_run(dir_path, raw=True, rng=None, format_version='0.5'):
    # Simulate reduced AGIPD data, with varying number of frames per train.
    # Counts across modules should be consistent
    prefix = 'RAW' if raw else 'CORR'
    if rng is None:
        rng = np.random.RandomState()

    frame_counts = rng.randint(0, 20, size=64)
    for modno in range(16):
        path = osp.join(dir_path,
                        '{}-R0238-AGIPD{:0>2}-S00000.h5'.format(prefix, modno))
        write_file(path, [
            AGIPDModule('SPB_DET_AGIPD1M-1/DET/{}CH0'.format(modno), raw=raw,
                         frames_per_train=frame_counts)
            ], ntrains=64, chunksize=32, format_version=format_version)

        if modno == 9 and not raw:
            # For testing masked_data
            with h5py.File(path, 'a') as f:
                mask_ds = f['INSTRUMENT/SPB_DET_AGIPD1M-1/DET/9CH0:xtdf/image/mask']
                mask_ds[0, 0, :32] = np.arange(32)

    write_file(osp.join(dir_path, '{}-R0238-DA01-S00000.h5'.format(prefix)),
               [ XGM('SA1_XTD2_XGM/DOOCS/MAIN'),
                 XGM('SPB_XTD9_XGM/DOOCS/MAIN'),
                 BaslerCam('SPB_IRU_CAM/CAM/SIDEMIC', sensor_size=(1024, 768))
               ], ntrains=32, chunksize=32, format_version=format_version)

    write_file(osp.join(dir_path, '{}-R0238-DA01-S00001.h5'.format(prefix)),
               [ XGM('SA1_XTD2_XGM/DOOCS/MAIN'),
                 XGM('SPB_XTD9_XGM/DOOCS/MAIN'),
                 BaslerCam('SPB_IRU_CAM/CAM/SIDEMIC', sensor_size=(1024, 768))
               ], ntrains=32, firsttrain=10032, chunksize=32,
               format_version=format_version)


def make_modern_spb_proc_run(dir_path, format_version='1.2'):
    for modno in range(16):
        path = osp.join(dir_path, f'CORR-R0142-AGIPD{modno:0>2}-S00000.h5')
        write_file(path, [
            AGIPDModule(f'SPB_DET_AGIPD1M-1/CORR/{modno}CH0', channel_name='output',
                        raw=False, frames_per_train=32,
                        legacy_name=f'SPB_DET_AGIPD1M-1/DET/{modno}CH0')
            ], ntrains=64, chunksize=32, format_version=format_version)

    # Ensure one chunk of mask data is actually written
    with h5py.File(path, 'r+') as f:
        ds = f['INSTRUMENT/SPB_DET_AGIPD1M-1/CORR/15CH0:output/image/mask']
        ds[0, 0, 5] = 1


def make_agipd1m_run(
    dir_path,
    rep_rate=True,
    gain_setting=True,
    integration_time=True,
    bias_voltage=True
):
    # Naming based on /gpfs/exfel/exp/SPB/202130/p900203/raw/r9015
    for modno in range(16):
        path = osp.join(dir_path, f'RAW-R9015-AGIPD{modno:02}-S00000.h5')
        write_file(path, [
            AGIPDModule(
                f'SPB_DET_AGIPD1M-1/DET/{modno}CH0', frames_per_train=64)
        ], ntrains=100, chunksize=32, format_version='1.0')

    write_file(osp.join(dir_path, 'RAW-R9015-AGIPD1MCTRL00-S00000.h5'), [
        AGIPDMDL(
              'SPB_IRU_AGIPD1M1/MDL/FPGA_COMP',
              rep_rate=rep_rate,
              gain_setting=gain_setting,
              integration_time=integration_time,
        ),
        AGIPD1MFPGA('SPB_IRU_AGIPD1M1/FPGA/MASTER_H1'),
        AGIPD1MPSC('SPB_IRU_AGIPD1M/PSC/HV', bias_voltage=bias_voltage),
     ], ntrains=100, chunksize=1, format_version='1.0')


def make_agipd500k_run(dir_path):
    # Naming based on /gpfs/exfel/exp/SPB/202130/p900203/raw/r9023
    for modno in range(8):
        path = osp.join(dir_path, f'RAW-R9023-AGIPD{modno:02}-S00000.h5')
        write_file(path, [
            AGIPDModule(
                f'HED_DET_AGIPD500K2G/DET/{modno}CH0', frames_per_train=64)
        ], ntrains=100, chunksize=32, format_version='1.0')

    write_file(osp.join(dir_path, 'RAW-R9023-AGIPD500K2G00-S00000.h5'), [
        AGIPDMDL('HED_EXP_AGIPD500K2G/MDL/FPGA_COMP'),
        AGIPD500KFPGA('HED_EXP_AGIPD500K2G/FPGA/M_0'),
     ], ntrains=100, chunksize=1, format_version='1.0')


def make_jungfrau_run(dir_path):
    # Naming based on /gpfs/exfel/exp/SPB/202022/p002732/raw/r0012
    for modno in range(1, 9):
        path = osp.join(dir_path, f'RAW-R0012-JNGFR{modno:02}-S00000.h5')
        write_file(path, [
            JUNGFRAUModule(f'SPB_IRDA_JF4M/DET/JNGFR{modno:02}')
        ], ntrains=100, chunksize=1, format_version='1.0')

    write_file(osp.join(dir_path, f'RAW-R0012-JNGFRCTRL00-S00000.h5'), [
        JUNGFRAUControl('SPB_IRDA_JF4M/DET/CONTROL'),
        JUNGFRAUMonitor('SPB_IRDA_JF4M/MDL/MONITOR'),
        JUNGFRAUPower('SPB_IRDA_JF4M/MDL/POWER'),
    ], ntrains=100, chunksize=1, format_version='1.0')

def make_fxe_jungfrau_run(dir_path):
    # Naming based on /gpfs/exfel/exp/FXE/202101/p002478/raw/
    for modno in range(1, 3):
        path = osp.join(dir_path, f'RAW-R0012-JNGFR{modno:02}-S00000.h5')
        write_file(path, [
            JUNGFRAUModule(f'FXE_XAD_JF1M/DET/JNGFR{modno:02}')
        ], ntrains=100, chunksize=1, format_version='1.0')

    path = osp.join(dir_path, f'RAW-R0052-JNGFR03-S00000.h5')
    write_file(path, [
        JUNGFRAUModule(f'FXE_XAD_JF500K/DET/JNGFR03')
    ], ntrains=100, chunksize=1, format_version='1.0')
    with h5py.File(path, 'a') as f:
        # For testing masked_data
        mask_ds = f['INSTRUMENT/FXE_XAD_JF500K/DET/JNGFR03:daqOutput/data/mask']
        mask_ds[0, 0, 0, :32] = np.arange(32)

    write_file(osp.join(dir_path, f'RAW-R0052-JNGFRCTRL00-S00000.h5'), [
        JUNGFRAUControl('FXE_XAD_JF1M/DET/CONTROL'),
        JUNGFRAUControl('FXE_XAD_JF500K/DET/CONTROL'),
    ], ntrains=100, chunksize=1, format_version='1.0')

def make_remi_run(dir_path):
    write_file(osp.join(dir_path, f'CORR-R0210-REMI01-S00000.h5'), [
        ReconstructedDLD6('SQS_REMI_DLD6/DET/TOP'),
    ], ntrains=100, chunksize=1, format_version='1.0')

def make_scs_run(dir_path):
    # Multiple sequence files for detector modules
    for modno in range(16):
        mod = DSSCModule(f'SCS_DET_DSSC1M-1/DET/{modno}CH0', frames_per_train=64)
        for seq in range(2):
            path = osp.join(dir_path, f'RAW-R0163-DSSC{modno:0>2}-S{seq:0>5}.h5')
            write_file(path, [mod], ntrains=64, firsttrain=(10000 + seq * 64),
                       chunksize=32, format_version='1.0')

if __name__ == '__main__':
    make_agipd_example_file('agipd_example.h5')
    make_fxe_da_file('fxe_control_example.h5')
    make_sa3_da_file('sa3_control_example.h5')
    make_agipd_file('agipd_example2.h5')
    make_lpd_file('lpd_example.h5')
    os.makedirs('fxe_example_run', exist_ok=True)
    make_fxe_run('fxe_example_run')
    os.makedirs('spb_example_run', exist_ok=True)
    make_spb_run('spb_example_run')
    print("Written examples.")
