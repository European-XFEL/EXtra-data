"""Test streaming data with ZMQ interface."""

import os

import pytest
from subprocess import PIPE, Popen

from extra_data.export import ZMQStreamer
from karabo_bridge import Client


# @pytest.fixture(scope='function')
# def file_server(mock_fxe_raw_run):
#     args = ['karabo-bridge-serve-files', str(mock_fxe_raw_run), str(3333)]
#     interface = ''

#     with Popen(args, stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True,
#                encoding='utf-8', env=dict(os.environ, PYTHONUNBUFFERED='1')
#                ) as p:
#         for line in p.stdout:
#             if line.startswith('Streamer started on:'):
#                 interface = line.partition(':')[2].strip()
#                 break
#         yield interface
#         p.kill()


# @pytest.fixture(scope='function')
# def file_server_with_combined_detector(mock_fxe_raw_run):
#     args = ['karabo-bridge-serve-files', str(mock_fxe_raw_run), str(3333),
#             '--append-detector-modules']
#     interface = ''

#     with Popen(args, stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True,
#                encoding='utf-8', env=dict(os.environ, PYTHONUNBUFFERED='1')
#                ) as p:
#         for line in p.stdout:
#             if line.startswith('Streamer started on:'):
#                 interface = line.partition(':')[2].strip()
#                 break
#         yield interface
#         p.kill()


def test_serve_files(mock_fxe_raw_run):
    args = ['karabo-bridge-serve-files', str(mock_fxe_raw_run), str(3333)]
    interface = ''

    with Popen(args, stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True,
               encoding='utf-8', env=dict(os.environ, PYTHONUNBUFFERED='1')
               ) as p:
        for line in p.stdout:
            if line.startswith('Streamer started on:'):
                interface = line.partition(':')[2].strip()
                break

        with Client(interface, timeout=5) as c:
            data, meta = c.next()

        assert 'FXE_DET_LPD1M-1/DET/0CH0:xtdf' in data
        assert data['FXE_DET_LPD1M-1/DET/0CH0:xtdf']['image.data'].shape == (128, 1, 256, 256)

        p.kill()


def test_serve_files_combined_detector(mock_fxe_raw_run):
    args = ['karabo-bridge-serve-files', str(mock_fxe_raw_run), str(3333),
            '--append-detector-modules']
    interface = ''

    with Popen(args, stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True,
               encoding='utf-8', env=dict(os.environ, PYTHONUNBUFFERED='1')
               ) as p:
        for line in p.stdout:
            if line.startswith('Streamer started on:'):
                interface = line.partition(':')[2].strip()
                break
        with Client(interface, timeout=5) as c:
            data, meta = c.next()

        assert 'FXE_DET_LPD1M-1/DET/APPEND' in data
        assert 'FXE_DET_LPD1M-1/DET/0CH0:xtdf' not in data
        assert data['FXE_DET_LPD1M-1/DET/APPEND']['image.data'].shape == (128, 1, 16, 256, 256)

        p.kill()

    
def test_deprecated_server():
    with pytest.deprecated_call():
        with ZMQStreamer(2222):
            pass


if __name__ == '__main__':
    pytest.main(["-v"])
    print("Run 'py.test -v -s' to see more output")
