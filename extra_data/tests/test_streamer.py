"""Test streaming data with ZMQ interface."""

import msgpack
import msgpack_numpy as numpack
import numpy as np
import pytest
from queue import Full
from subprocess import Popen

from extra_data.export import ZMQStreamer
from karabo_bridge import Client


@pytest.fixture(scope='function')
def server():
    with ZMQStreamer(2222, maxlen=10) as s:
        yield s


@pytest.fixture(scope='function')
def file_server(mock_fxe_raw_run):
    port = 3333
    args = [
        'karabo-bridge-serve-files', str(mock_fxe_raw_run), str(port),
        '--append-detector-modules'
    ]
    p = Popen(args)
    yield f'tcp://localhost:{port}'
    p.kill()


@pytest.mark.parametrize('protocol_version', ['1.0', '2.2'])
def test_fill_queue(server):
    for i in range(10):
        server.feed({str(i): {str(i): i}})

    assert server.buffer.full()
    with pytest.raises(Full):
        server.feed({'too much': {'prop': 0}}, block=False)


def test_req_rep(server):
    client = Client(server.endpoint)
    data = {'a': {'b': 1}}

    for _ in range(3):
        server.feed(data)

    for _ in range(3):
        d, metadata = client.next()
        assert d == data


def test_serve_files(file_server):
    with Client(file_server, timeout=1) as c:
        data, meta = c.next()

    assert 'FXE_DET_LPD1M-1/DET/APPEND' in data
    assert 'FXE_DET_LPD1M-1/DET/0CH0:xtdf' not in data
    assert data['FXE_DET_LPD1M-1/DET/APPEND']['image.data'].shape == (128, 1, 16, 256, 256)


def test_serve_files(file_server):
    with Client(file_server, timeout=1) as c:
        data, meta = c.next()

    assert 'FXE_DET_LPD1M-1/DET/APPEND' in data
    assert 'FXE_DET_LPD1M-1/DET/0CH0:xtdf' not in data
    assert data['FXE_DET_LPD1M-1/DET/APPEND']['image.data'].shape == (128, 1, 16, 256, 256)


if __name__ == '__main__':
    pytest.main(["-v"])
    print("Run 'py.test -v -s' to see more output")
