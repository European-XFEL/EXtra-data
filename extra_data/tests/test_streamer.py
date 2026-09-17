"""Test streaming data with ZMQ interface."""

import os
import time
import select
import signal
import dataclasses
from subprocess import PIPE, Popen, TimeoutExpired
from unittest import mock

import pytest
import numpy as np
import xarray as xr

from extra_data import open_run, by_id, H5File, RunDirectory
from extra_data.export import _iter_trains, ZMQStreamer, MemoryStreamer
from karabo_bridge import Client


def test_merge_detector(mock_fxe_raw_run, mock_fxe_control_data, mock_spb_proc_run):
    with RunDirectory(mock_fxe_raw_run) as run:
        for tid, data in _iter_trains(run, merge_detector=True):
            assert 'FXE_DET_LPD1M-1/DET/APPEND' in data
            assert 'FXE_DET_LPD1M-1/DET/0CH0:xtdf' not in data
            shape = data['FXE_DET_LPD1M-1/DET/APPEND']['image.data'].shape
            assert shape == (128, 1, 16, 256, 256)
            break

        for tid, data in _iter_trains(run):
            assert 'FXE_DET_LPD1M-1/DET/0CH0:xtdf' in data
            shape = data['FXE_DET_LPD1M-1/DET/0CH0:xtdf']['image.data'].shape
            assert shape == (128, 1, 256, 256)
            break

    with H5File(mock_fxe_control_data) as run:
        for tid, data in _iter_trains(run, merge_detector=True):
            assert frozenset(data) == run.select_trains(by_id[[tid]]).all_sources
            break

    with RunDirectory(mock_spb_proc_run) as run:
        for tid, data in _iter_trains(run, merge_detector=True):
            shape = data['SPB_DET_AGIPD1M-1/DET/APPEND']['image.data'].shape
            assert shape == (64, 16, 512, 128)
            shape = data['SPB_DET_AGIPD1M-1/DET/APPEND']['image.gain'].shape
            assert shape == (64, 16, 512, 128)
            shape = data['SPB_DET_AGIPD1M-1/DET/APPEND']['image.mask'].shape
            assert shape == (64, 16, 512, 128)
            break


def cleanup_proc(p: Popen):
    if p.poll() is None:
        p.send_signal(signal.SIGINT)
        try:
            p.wait(timeout=2)
        except TimeoutExpired:
            pass
    if p.poll() is None:
        p.kill()
        rc = p.wait(timeout=2)
        assert rc == -9  # process terminated by kill signal


@pytest.mark.skipif(os.name != 'posix', reason="Test uses Unix socket")
def test_serve_files(mock_fxe_raw_run, tmp_path):
    src = 'FXE_XAD_GEC/CAM/CAMERA:daqOutput'
    args = ['karabo-bridge-serve-files', '-z', 'PUSH', str(mock_fxe_raw_run),
            f'ipc://{tmp_path}/socket', '--source', src]
    interface = None

    p = Popen(args, stdin=PIPE, stdout=PIPE, stderr=PIPE,
               env=dict(os.environ, PYTHONUNBUFFERED='1'))
    try:
        for line in p.stdout:
            line = line.decode('utf-8')
            if line.startswith('Streamer started on:'):
                interface = line.partition(':')[2].strip()
                break

        print('interface:', interface)
        assert interface is not None, p.stderr.read().decode()

        with Client(interface, sock='PULL', timeout=30) as c:
            data, meta = c.next()

        tid = next(m['timestamp.tid'] for m in meta.values())
        assert tid == 10000
        assert set(data) == {src}
    finally:
        cleanup_proc(p)


@pytest.mark.skipif(os.name != 'posix', reason="Test uses Unix socket")
def test_serve_run(mock_spb_raw_and_proc_run, tmp_path):
    mock_data_root, _, _ = mock_spb_raw_and_proc_run
    zmq_endpoint = f'ipc://{tmp_path}/socket'
    xgm_src = 'SPB_XTD9_XGM/DOOCS/MAIN'
    agipd_m0_src = 'SPB_DET_AGIPD1M-1/DET/0CH0:xtdf'
    args = ['karabo-bridge-serve-run', '2012', '238',
            '--port', zmq_endpoint,
            '--include', f'{xgm_src}[beamPosition.i*Pos]',
            '--include', '*AGIPD1M-1/DET/0CH0:xtdf'
           ]

    p = Popen(args, env=dict(
        os.environ,
        PYTHONUNBUFFERED='1',
    ))
    try:
        with Client(zmq_endpoint, timeout=30) as c:
            data, meta = c.next()

        tid = next(m['timestamp.tid'] for m in meta.values())
        assert tid == 10000
        assert set(data) == {xgm_src, agipd_m0_src}
        assert set(data[xgm_src]) == \
               {f'beamPosition.i{xy}Pos.value' for xy in 'xy'} | {'metadata'}
        assert data[agipd_m0_src]['image.data'].dtype == np.float32
    finally:
        cleanup_proc(p)


def test_deprecated_server():
    with pytest.deprecated_call():
        with ZMQStreamer(2222):
            pass


def test_memorystreamer(mock_spb_raw_and_proc_run, monkeypatch):
    root_dir, raw_dir, proc_dir = mock_spb_raw_and_proc_run

    run = open_run(2012, 238)
    photon_flux = run["SPB_XTD9_XGM/DOOCS/MAIN", "pulseEnergy.photonFlux"]
    camera = run["SPB_IRU_CAM/CAM/SIDEMIC:daqOutput", "data.image.pixels"]
    xgm = run["SPB_XTD9_XGM/DOOCS/MAIN:output", "data.intensityTD"]

    xgm_data = xgm.xarray()
    camera_data = camera.xarray()
    photon_flux_data = photon_flux.xarray()

    streamer = MemoryStreamer()

    for train_data in streamer.iterate(dict(camera=camera, xgm=xgm, photon_flux=photon_flux)):
        # Check that we have all the expected fields in the custom dataclass
        field_names = [x.name for x in dataclasses.fields(train_data)]
        assert set(field_names) == {"trainId", "idx", "_chunk", "_key_slots", "camera", "xgm", "photon_flux"}

        # With the default settings we should be iterating train-by-train,
        # so check that the trainId field is integral.
        assert train_data.trainId == run.train_ids[0]
        assert train_data.idx == 0

        # Smoke test to check if the arrays and their metadata are the same
        xr.testing.assert_identical(train_data.camera, camera_data.sel(trainId=train_data.trainId))
        xr.testing.assert_identical(train_data.xgm, xgm_data.sel(trainId=train_data.trainId))
        xr.testing.assert_identical(train_data.photon_flux, photon_flux_data.sel(trainId=train_data.trainId))

        # The slow data should have been cached, under the source/key rather
        # than the label the user gave it.
        assert [key[:2] for key in streamer._cache] == \
            [(photon_flux.source, photon_flux.key)]

        # Exiting early tests that the generators resources are cleaned up
        # properly when triggered by the GC (see the comments in .iterate()).
        break

    # Test that autosqueeze can be disabled
    for train_data in streamer.iterate(dict(camera=camera, photon_flux=photon_flux), autosqueeze=False):
        assert len(train_data.trainId) == 1
        assert train_data.trainId[0] == run.train_ids[0]

        assert len(train_data.idx) == 1
        assert train_data.idx[0] == 0

        assert train_data.camera.shape[0] == 1
        assert train_data.photon_flux.shape[0] == 1

        break

    # Test that we iterate over the trains in order
    tids = []
    idxs = []
    for train_data in streamer.iterate(dict(camera=camera)):
        tids.append(train_data.trainId)
        idxs.append(train_data.idx)
    assert tids == run.train_ids
    assert idxs == list(range(len(run.train_ids)))

    # Choosing incompatible part sizes should throw an exception
    with pytest.raises(ValueError):
        for _ in streamer.iterate(dict(camera=camera), trains_per_part=2, reader_trains_per_chunk=3):
            pass

    # Test iterating with different part sizes
    tids = []
    idxs = []
    for train_data in streamer.iterate(dict(camera=camera),
                                       trains_per_part=4, reader_trains_per_chunk=12):
        assert len(train_data.trainId) <= 4
        assert len(train_data.trainId) == len(train_data.idx)
        tids.extend(train_data.trainId)
        idxs.extend(train_data.idx)
    assert tids == run.train_ids
    assert idxs == list(range(len(run.train_ids)))

    # Test alignment
    tids = []
    idxs = []
    for train_data in streamer.iterate(dict(camera=camera[10:20], xgm=xgm)):
        tids.append(train_data.trainId)
        idxs.append(train_data.idx)
    assert tids == run.train_ids[10:20]
    assert idxs == list(range(10))

    # Smoke test using minimal slots and workers to see if there are any
    # hangs/concurrency issues.
    streamer = MemoryStreamer(pool_size=1)
    for train_data in streamer.iterate(dict(camera=camera), n_buffer_slots=1):
        pass


def test_memorystreamer_stop_early(mock_spb_raw_and_proc_run):
    run = open_run(2012, 238)
    camera = run["SPB_IRU_CAM/CAM/SIDEMIC:daqOutput", "data.image.pixels"]

    # Stopping early must not deadlock the producer. With fewer slots than
    # chunks the submitter threads are still waiting for slots when we stop, so
    # they never submit the remaining chunks.
    streamer = MemoryStreamer(pool_size=2)
    for train_data in streamer.iterate(dict(camera=camera),
                                       reader_trains_per_chunk=1, n_buffer_slots=1):
        break

    streamer.close()


def test_memorystreamer_multiple_entries(mock_spb_raw_run):
    run = RunDirectory(mock_spb_raw_run)
    module = run["SPB_DET_AGIPD1M-1/DET/0CH0:xtdf", "image.data"]
    camera = run["SPB_IRU_CAM/CAM/SIDEMIC:daqOutput", "data.image.pixels"]

    # Each train of the detector holds 64 frames, which get an axis of their
    # own instead of being folded into the train axis.
    frames_per_train = 64
    reference = module.ndarray().reshape(len(module.train_ids), frames_per_train,
                                         *module.entry_shape)

    streamer = MemoryStreamer()

    for i, train_data in enumerate(streamer.iterate(dict(module=module, camera=camera))):
        assert train_data.module.dims == ("entry", "dim_0", "dim_1", "dim_2")
        np.testing.assert_array_equal(train_data.module.values, reference[i])

        # A source with one entry per train has no entry axis
        assert train_data.camera.dims == ("dim_0", "dim_1")

    # Parts of several trains keep both axes, and the entries stay with their
    # own train.
    n_trains = 0
    for train_data in streamer.iterate(dict(module=module),
                                       trains_per_part=4, reader_trains_per_chunk=12):
        assert train_data.module.dims == ("trainId", "entry", "dim_0", "dim_1", "dim_2")
        assert train_data.module.shape[:2] == (len(train_data.trainId), frames_per_train)
        np.testing.assert_array_equal(train_data.module.values,
                                      reference[n_trains:n_trains + len(train_data.trainId)])
        assert list(train_data.module.coords["trainId"].values) == list(train_data.trainId)
        n_trains += len(train_data.trainId)

    assert n_trains == len(module.train_ids)
    streamer.close()


def test_memorystreamer_irregular_entries(mock_reduced_spb_raw_run):
    run = RunDirectory(mock_reduced_spb_raw_run)
    module = run["SPB_DET_AGIPD1M-1/DET/0CH0:xtdf", "image.data"]
    assert len(np.unique(module.data_counts(labelled=False))) > 1

    # A slot is a rectangle, so trains with differing numbers of entries can't
    # be streamed.
    streamer = MemoryStreamer()
    with pytest.raises(ValueError, match="different numbers of entries"):
        for _ in streamer.iterate(dict(module=module)):
            pass

    streamer.close()


def test_memorystreamer_bad_sources(mock_spb_raw_and_proc_run):
    run = open_run(2012, 238)
    camera = run["SPB_IRU_CAM/CAM/SIDEMIC:daqOutput", "data.image.pixels"]
    streamer = MemoryStreamer()

    # Passing something that isn't a KeyData
    with pytest.raises(TypeError):
        for _ in streamer.iterate(dict(camera=camera.ndarray())):
            pass

    # Passing no sources at all
    with pytest.raises(ValueError):
        for _ in streamer.iterate(dict()):
            pass


def test_memorystreamer_eventfd(mock_spb_raw_and_proc_run):
    run = open_run(2012, 238)
    camera = run["SPB_IRU_CAM/CAM/SIDEMIC:daqOutput", "data.image.pixels"]
    camera_data = camera.xarray()

    def wait(iterator):
        # Stand-in for whatever the consumer's event loop uses to wait for the
        # fd to become readable (e.g. Julia's poll_fd()).
        r, _, _ = select.select([iterator.fd], [], [], 10)
        assert r, "Timed out waiting for the streamer"

    streamer = MemoryStreamer()

    # Test that starting and stopping the streamer immediately works
    iterator = streamer.event_iterator(dict(camera=camera))
    iterator.close()

    # Test consuming the whole stream, returning slots as we go. The fd is only
    # a wake-up, so everything that's ready is drained after each one.
    iterator = streamer.event_iterator(dict(camera=camera))
    tids = []
    while not iterator.ended:
        wait(iterator)
        while (train_data := iterator.get()) is not None:
            xr.testing.assert_identical(train_data.camera,
                                        camera_data.sel(trainId=train_data.trainId))
            tids.append(train_data.trainId)
            iterator.done(train_data)

    assert tids == run.train_ids

    # Polling again after the end keeps returning None rather than raising
    assert iterator.get() is None

    iterator.close()

    # Test consuming a finite number of items and then stopping early
    iterator = streamer.event_iterator(dict(camera=camera))
    wait(iterator)
    assert iterator.get().trainId == run.train_ids[0]
    iterator.close()

    # Consume an entire dataset without returning any slots to the producer.
    # This only works if there are as many slots as trains, otherwise the
    # producer would block waiting for a free slot.
    camera_sel = camera[:5]
    iterator = streamer.event_iterator(dict(camera=camera_sel),
                                       reader_trains_per_chunk=1,
                                       n_buffer_slots=len(camera_sel.train_ids))
    n_loaded = 0
    while n_loaded < len(camera_sel.train_ids):
        wait(iterator)
        while iterator.get() is not None:
            n_loaded += 1
    iterator.close()
    assert n_loaded == len(camera_sel.train_ids)


def test_memorystreamer_eventfd_error(mock_spb_raw_and_proc_run):
    """A failure in the producer must wake the consumer, not hang it."""
    run = open_run(2012, 238)
    camera = run["SPB_IRU_CAM/CAM/SIDEMIC:daqOutput", "data.image.pixels"]

    streamer = MemoryStreamer()
    # trains_per_part not dividing reader_trains_per_chunk makes _stream() raise
    iterator = streamer.event_iterator(dict(camera=camera),
                                       trains_per_part=2,
                                       reader_trains_per_chunk=3)
    r, _, _ = select.select([iterator.fd], [], [], 30)
    assert r, "Producer died without waking the consumer"
    with pytest.raises(ValueError):
        iterator.get()


if __name__ == '__main__':
    pytest.main(["-v"])
    print("Run 'py.test -v -s' to see more output")
