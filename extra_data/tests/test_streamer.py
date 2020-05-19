"""Test streaming data with ZMQ interface."""

import os

import pytest
import signal
from subprocess import PIPE, Popen

from extra_data import by_id, H5File, RunDirectory
from extra_data.export import _iter_trains, ZMQStreamer
from karabo_bridge import Client


def test_merge_detector(mock_fxe_raw_run, mock_fxe_control_data):
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


def test_subproc():
    args = ['sleep', '3']
    with Popen(args, stdin=PIPE, stdout=PIPE, stderr=PIPE,
               env=dict(os.environ, PYTHONUNBUFFERED='1')) as p:
        p.kill()


def test_long_subproc():
    args = ['sleep', '999999999']
    with Popen(args, stdin=PIPE, stdout=PIPE, stderr=PIPE,
               env=dict(os.environ, PYTHONUNBUFFERED='1')) as p:
        p.kill()


def test_serve_files(mock_fxe_raw_run):
    args = ['karabo-bridge-serve-files', str(mock_fxe_raw_run), str(44444)]
    interface = ''

    with Popen(args, stdin=PIPE, stdout=PIPE, stderr=PIPE,
               env=dict(os.environ, PYTHONUNBUFFERED='1')) as p:
        for line in p.stdout:
            line = line.decode('utf-8')
            if line.startswith('Streamer started on:'):
                interface = line.partition(':')[2].strip()
                break

        with Client(interface, timeout=5) as c:
            data, meta = c.next()

        tid = next(m['timestamp.tid'] for m in meta.values())
        sources = RunDirectory(mock_fxe_raw_run).select_trains(by_id[[tid]]).all_sources
        assert frozenset(data) == sources

        p.send_signal(signal.SIGINT)
        p.communicate(timeout=2)

        # p.kill()
        # rc = p.wait(timeout=2)
        # assert rc == -9  # process terminated by kill signal


def test_deprecated_server():
    with pytest.deprecated_call():
        with ZMQStreamer(2222):
            pass


if __name__ == '__main__':
    pytest.main(["-v"])
    print("Run 'py.test -v -s' to see more output")
