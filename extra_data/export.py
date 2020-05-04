# coding: utf-8
"""Expose data to different interface

ZMQStream explose to a ZeroMQ socket in a REQ/REP pattern.

Copyright (c) 2017, European X-Ray Free-Electron Laser Facility GmbH
All rights reserved.

You should have received a copy of the 3-Clause BSD License along with this
program. If not, see <https://opensource.org/licenses/BSD-3-Clause>
"""

from argparse import ArgumentParser
import os.path as osp
from warnings import warn

from karabo_bridge import ServerInThread

from .reader import RunDirectory, H5File
from .stacking import stack_detector_data
from .utils import find_infiniband_ip

__all__ = ['ZMQStreamer', 'serve_files']


class ZMQStreamer(ServerInThread):
    def __init__(self, port, sock='REP', maxlen=10, protocol_version='2.2',
                 dummy_timestamps=False):
        warn("Please use :ref:karabo_bridge.ServerInThread instead",
             DeprecationWarning, stacklevel=2)

        endpoint = f'tcp://*:{port}'
        super().__init__(endpoint, sock='REP', maxlen=10,
                         protocol_version='2.2', dummy_timestamps=False)


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

    endpoint = f'tcp://{find_infiniband_ip() if use_infiniband else "*"}:{port}'
    streamer = ServerInThread(endpoint, dummy_timestamps=dummy_timestamps)
    streamer.start()
    print(f'Streamer started on: {streamer.endpoint}')
    for tid, train_data in data.trains():
        if train_data:
            if append_detector_modules:
                if source_glob == '*':
                    warn("You are trying to stack detector-module sources"
                         "with a global wildcard (\'*\'). If there are non-"
                         "detector sources in your run, this will fail.")
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
