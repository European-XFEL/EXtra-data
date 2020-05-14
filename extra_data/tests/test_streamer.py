"""Test streaming data with ZMQ interface."""

import pytest
from subprocess import Popen

from extra_data.export import ZMQStreamer
from karabo_bridge import Client


# @pytest.fixture(scope='function')
# def file_server(mock_fxe_raw_run):
#     port = 3333
#     args = ['karabo-bridge-serve-files', str(mock_fxe_raw_run), str(port)]
#     p = Popen(args)
#     yield f'tcp://127.0.0.1:{port}'
#     p.kill()


@pytest.fixture(scope='function')
def file_server_with_combined_detector(mock_fxe_raw_run):
    port = 3333
    args = [
        'karabo-bridge-serve-files', str(mock_fxe_raw_run), str(port),
        '--append-detector-modules'
    ]
    p = Popen(args)
    yield f'tcp://127.0.0.1:{port}'
    p.kill()


def test_serve_files(mock_fxe_raw_run):  # file_server):
    port = 3333
    args = ['karabo-bridge-serve-files', str(mock_fxe_raw_run), str(port)]
    p = Popen(args)
    from time import sleep

    with Client(f'tcp://127.0.0.1:{port}', timeout=5) as c:
        data, meta = c.next()

    assert 'FXE_DET_LPD1M-1/DET/0CH0:xtdf' in data
    assert data['FXE_DET_LPD1M-1/DET/0CH0:xtdf']['image.data'].shape == (128, 1, 256, 256)
    p.kill()


def test_serve_files_combined_detector(file_server_with_combined_detector):
    with Client(file_server_with_combined_detector, timeout=5) as c:
        data, meta = c.next()

    assert 'FXE_DET_LPD1M-1/DET/APPEND' in data
    assert 'FXE_DET_LPD1M-1/DET/0CH0:xtdf' not in data
    assert data['FXE_DET_LPD1M-1/DET/APPEND']['image.data'].shape == (128, 1, 16, 256, 256)

    
def test_deprecated_server():
    with pytest.deprecated_call():
        with ZMQStreamer(2222):
            pass


if __name__ == '__main__':
    pytest.main(["-v"])
    print("Run 'py.test -v -s' to see more output")
