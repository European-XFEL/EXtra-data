"""Test streaming data with ZMQ interface."""

import pytest

from extra_data.export import ZMQStreamer
from karabo_bridge import Client


# @pytest.fixture(scope='function')
# def server():
#     with ZMQStreamer(2222, maxlen=10) as s:
#         yield s


# def test_req_rep(server):
#     client = Client(server.endpoint)
#     data = {'a': {'b': 1}}

#     for _ in range(3):
#         server.feed(data)

#     for _ in range(3):
#         d, metadata = client.next()
#         assert d == data


def test_deprecated_server():
    with pytest.deprecated_call():
        server = ZMQStreamer(2222)


if __name__ == '__main__':
    pytest.main(["-v"])
    print("Run 'py.test -v -s' to see more output")
