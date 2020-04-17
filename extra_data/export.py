# coding: utf-8
"""Expose data to different interface

ZMQStream explose to a ZeroMQ socket in a REQ/REP pattern.

Copyright (c) 2017, European X-Ray Free-Electron Laser Facility GmbH
All rights reserved.

You should have received a copy of the 3-Clause BSD License along with this
program. If not, see <https://opensource.org/licenses/BSD-3-Clause>
"""

from functools import partial
from argparse import ArgumentParser
import os.path as osp
from queue import Empty, Queue
from socket import gethostname
from threading import Thread
from warnings import warn

from karabo_bridge import serialize
import zmq

from .reader import RunDirectory, H5File
from .stacking import stack_detector_data
from .utils import find_infiniband_ip

__all__ = ['ZMQStreamer', 'serve_files']


class ZMQStreamer(Thread):
    def __init__(self, port, sock='REP', maxlen=10, protocol_version='2.2',
                 dummy_timestamps=False, use_infiniband=False):
        """ZeroMQ interface sending data over a TCP socket.

        example::

            # Server:
            serve = ZMQStreamer(1234)
            serve.start()

            for tid, data in run.trains():
                result = important_processing(data)
                serve.feed(result)

            # Client:
            from karabo_bridge import Client
            client = Client('tcp://server.hostname:1234')
            data = client.next()

        Parameters
        ----------
        port: int
            Local TCP port to bind socket to
        maxlen: int, optional
            How many trains to cache before sending (default: 10)
        protocol_version: ('1.0' | '2.1')
            Which version of the bridge protocol to use. Defaults to the latest
            version implemented.
        dummy_timestamps: bool
            Some tools (such as OnDA) expect the timestamp information to be in
            the messages. We can't give accurate timestamps where these are not
            in the file, so this option generates fake timestamps from the time
            the data is fed in.
        use_infiniband: bool
            Use infiniband interface if the host has one (default False)
        """
        super().__init__()
        self.serialize = partial(serialize, protocol_version=protocol_version,
                                 dummy_timestamps=dummy_timestamps)

        self.zmq_context = zmq.Context()
        if sock == 'REP':
            self.server_socket = self.zmq_context.socket(zmq.REP)
        elif sock == 'PUB':
            self.server_socket = self.zmq_context.socket(zmq.PUB)
        elif sock == 'PUSH':
            self.server_socket = self.zmq_context.socket(zmq.PUSH)
        else:
            raise ValueError(f'Unsupported socket type: {sock}')
        self.server_socket.setsockopt(zmq.LINGER, 0)
        self.server_socket.set_hwm(1)
        self.server_socket.bind(f'tcp://{find_infiniband_ip()}:{port}')

        self.stopper_r = self.zmq_context.socket(zmq.PAIR)
        self.stopper_r.bind('inproc://sim-server-stop')
        self.stopper_w = self.zmq_context.socket(zmq.PAIR)
        self.stopper_w.connect('inproc://sim-server-stop')

        self.buffer = Queue(maxsize=maxlen)

    @property
    def endpoint(self):
        endpoint = self.server_socket.getsockopt_string(zmq.LAST_ENDPOINT)
        endpoint = endpoint.replace('0.0.0.0', gethostname())
        return endpoint

    def run(self):
        poller = zmq.Poller()
        poller.register(self.server_socket, zmq.POLLIN | zmq.POLLOUT)
        poller.register(self.stopper_r, zmq.POLLIN)

        while True:
            events = dict(poller.poll())

            if self.stopper_r in events:
                self.stopper_r.recv()
                break

            try:
                payload = self.buffer.get(timeout=0.1)
            except Empty:
                continue

            if events[self.server_socket] is zmq.POLLIN:
                msg = self.server_socket.recv()
                if msg != b'next':
                    print(f'Unrecognised request: {msg}')
                    self.server_socket.send(b'Error: bad request %b' % msg)
                    continue

            self.server_socket.send_multipart(payload, copy=False)

    def feed(self, data, metadata=None, block=True, timeout=None):
        """Push data to the sending queue.

        This blocks if the queue already has *maxlen* items waiting to be sent.

        Parameters
        ----------
        data : dict
            Contains train data. The dictionary has to follow the karabo_bridge
            see :func:`~karabo_bridge.serializer.serialize` for details

        metadata : dict, optional
            Contains train metadata. If the metadata dict is not provided it
            will be extracted from 'data' or an empty dict if 'metadata' key
            is missing from a data source.
            see :func:`~karabo_bridge.serializer.serialize` for details

        block: bool
            If True, block if necessary until a free slot is available or
            'timeout' has expired. If False and there is no free slot, raises
            'queue.Full' (timeout is ignored)

        timeout: float
            In seconds, raises 'queue.Full' if no free slow was available
            within that time.
        """
        self.buffer.put(self.serialize(data, metadata),
                        block=block, timeout=timeout)

    def stop(self):
        if self.is_alive():
            self.stopper_w.send(b'')
            self.join()
        self.zmq_context.destroy(linger=0)

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()


def serve_files(path, port, source_glob='*', key_glob='*',
                append_detector_modules=False, dummy_timestamps=False,
                use_infiniband=False):
    """Stream data from files through a TCP socket.

    Parameters
    ----------
    path: str
        Path to the HDF5 file or file folder.
    port: int
        Local TCP port to bind socket to.
    source_glob: str
        Only stream sources matching this glob pattern.
        Streaming data selectively is more efficient than streaming everything.
    key_glob: str
        Only stream keys matching this glob pattern in the selected sources.
    append_detector_modules: bool
        Whether to combine module sources by stacking.
    dummy_timestamps: bool
        Whether to add mock timestamps if the metadata lacks them.
    use_infiniband: bool
        Use infiniband interface if available
    """
    if osp.isdir(path):
        data = RunDirectory(path)
    else:
        data = H5File(path)

    data = data.select(source_glob, key_glob)

    streamer = ZMQStreamer(port, dummy_timestamps=dummy_timestamps,
                           use_infiniband=use_infiniband)
    streamer.start()
    print(f'Streamer started on: {streamer.endpoint}')
    for tid, train_data in data.trains():
        if train_data:
            if append_detector_modules:
                if source_glob == '*':
                    warn(" You are trying to stack detector-module sources"
                    " with a global wildcard (\'*\'). If there are non-"
                    " detector sources in your run, this will fail.")
                stacked = stack_detector_data(train_data, 'image.data')
                merged_data = {}
                # the data key pretends this is online data from SPB
                merged_data['SPB_DET_AGIPD1M-1/CAL/APPEND_CORRECTED'] = {
                    'image.data': stacked
                }
                # sec, frac = gen_time() # use time-stamps from file data?
                metadata = {}
                metadata['SPB_DET_AGIPD1M-1/CAL/APPEND_CORRECTED'] = {
                    'source': 'SPB_DET_AGIPD1M-1/CAL/APPEND_CORRECTED',
                    'timestamp.tid': tid
                }
                streamer.feed(merged_data, metadata)
            else:
                streamer.feed(train_data)

    streamer.stop()


def main(argv=None):
    ap = ArgumentParser(prog="karabo-bridge-serve-files")
    ap.add_argument("path", help="Path of a file or run directory to serve")
    ap.add_argument("port", help="TCP port to run server on")
    ap.add_argument(
        "--source", help="Stream only matching sources ('*' is a wildcard)",
        default='*',
    )
    ap.add_argument(
        "--key", help="Stream only matching keys ('*' is a wildcard)",
        default='*',
    )
    ap.add_argument(
        "--append-detector-modules", help="combine multiple module sources"
        " into one (will only work for AGIPD data currently).",
        action='store_true'
    )
    ap.add_argument(
        "--dummy-timestamps", help="create dummy timestamps if the meta-data"
        " lacks proper timestamps",
        action='store_true'
    )
    ap.add_argument(
        "--use-infiniband", help="Use infiniband interface if available",
        action='store_true'
    )
    args = ap.parse_args(argv)

    serve_files(
        args.path, args.port, source_glob=args.source, key_glob=args.key,
        append_detector_modules=args.append_detector_modules,
        dummy_timestamps=args.dummy_timestamps,
        use_infiniband=args.use_infiniband
    )
