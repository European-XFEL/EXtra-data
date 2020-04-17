"""Test streaming data with ZMQ interface."""

import msgpack
import msgpack_numpy as numpack
import numpy as np
import pytest
from queue import Full

from extra_data.export import ZMQStreamer
from karabo_bridge import Client


DATA = {
    'source1': {
        'parameter.1.value': 123,
        'list.of.int': [1, 2, 3],
        'string.param': 'True',
        'boolean': False,
        'metadata': {'timestamp.tid': 9876543210},
    },
    'XMPL/DET/MOD0': {
        'image.data': np.random.randint(255, size=(2, 3, 4), dtype=np.uint8),
        'something.else': ['a', 'bc', 'd'],
    },
}


def compare_nested_dict(d1, d2, path=''):
    for key in d1.keys():
        if key not in d2:
            print(d1.keys())
            print(d2.keys())
            raise KeyError('key is missing in d2: {}{}'.format(path, key))

        if isinstance(d1[key], dict):
            path += key + '.'
            compare_nested_dict(d1[key], d2[key], path)
        else:
            v1 = d1[key]
            v2 = d2[key]

            try:
                if isinstance(v1, np.ndarray):
                    assert (v1 == v2).all()
                elif isinstance(v1, tuple) or isinstance(v2, tuple):
                    # msgpack doesn't know about complex types, everything is
                    # an array. So tuples are packed as array and then
                    # unpacked as list by default.
                    assert list(v1) == list(v2)
                else:
                    assert v1 == v2
            except AssertionError:
                raise ValueError('diff: {}{}'.format(path, key), v1, v2)


@pytest.fixture(scope='function')
def server(protocol_version):
    with ZMQStreamer(2222, maxlen=10, protocol_version=protocol_version) as s:
        yield s


@pytest.mark.parametrize('protocol_version', ['1.0', '2.2'])
def test_fill_queue(server):
    for i in range(10):
        server.feed({str(i): {str(i): i}})

    assert server.buffer.full()
    with pytest.raises(Full):
        server.feed({'too much': {'prop': 0}}, block=False)


@pytest.mark.parametrize('protocol_version', ['1.0', '2.2'])
def test_req_rep(server):
    client = Client(server.endpoint)

    for _ in range(3):
        server.feed(DATA)

    for _ in range(3):
        data, metadata = client.next()
        compare_nested_dict(DATA, data)


if __name__ == '__main__':
    pytest.main(["-v"])
    print("Run 'py.test -v -s' to see more output")
