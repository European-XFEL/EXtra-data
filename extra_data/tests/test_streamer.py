"""Test streaming data with ZMQ interface."""

import pytest

from extra_data.export import ZMQStreamer
from karabo_bridge import Client


def test_deprecated_server():
    with pytest.deprecated_call():
        server = ZMQStreamer(2222)


if __name__ == '__main__':
    pytest.main(["-v"])
    print("Run 'py.test -v -s' to see more output")
