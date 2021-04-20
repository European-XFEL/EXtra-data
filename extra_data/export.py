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
from karabo_bridge.server import Sender

from .components import XtdfDetectorBase
from .exceptions import SourceNameError
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
        super().__init__(endpoint, sock=sock, maxlen=maxlen,
                         protocol_version=protocol_version,
                         dummy_timestamps=dummy_timestamps)


def _iter_trains(data, merge_detector=False):
    """Iterate over trains in data and merge detector tiles in a single source

    :data: DataCollection
    :merge_detector: bool
        if True and data contains detector data (e.g. AGIPD) idividual sources
        for each detector tiles are merged in a single source. The new source
        name keep the original prefix, but replace the last 2 part with
        '/DET/APPEND'. Individual sources are removed from the train data

    :yield: dict
        train data
    """
    det, source_name = None, ''
    if merge_detector:
        for detector in XtdfDetectorBase.__subclasses__():
            try:
                det = detector(data)
                source_name = f'{det.detector_name}/DET/APPEND'
            except SourceNameError:
                continue
            else:
                break

    for tid, train_data in data.trains():
        if not train_data:
            continue

        if det is not None:
            det_data = {
                k: v for k, v in train_data.items()
                if k in det.data.detector_sources
            }

            # get one of the module to reference other datasets
            train_data[source_name] = mod_data = next(iter(det_data.values()))

            stacked = stack_detector_data(det_data, 'image.data')
            mod_data['image.data'] = stacked
            mod_data['metadata']['source'] = source_name

            if 'image.gain' in mod_data:
                stacked = stack_detector_data(det_data, 'image.gain')
                mod_data['image.gain'] = stacked
            if 'image.mask' in mod_data:
                stacked = stack_detector_data(det_data, 'image.mask')
                mod_data['image.mask'] = stacked

            # remove individual module sources
            for src in det.data.detector_sources:
                del train_data[src]

        yield tid, train_data


def serve_files(path, port, source_glob='*', key_glob='*',
                append_detector_modules=False, dummy_timestamps=False,
                use_infiniband=False, sock='REP'):
    """Stream data from files through a TCP socket.

    Parameters
    ----------
    path: str
        Path to the HDF5 file or file folder.
    port: str or int
        A ZMQ endpoint (e.g. 'tcp://*:44444') or a TCP port to bind the socket
        to. Integers or strings of all digits are treated as port numbers.
    source_glob: str
        Only stream sources matching this glob pattern.
        Streaming data selectively is more efficient than streaming everything.
    key_glob: str
        Only stream keys matching this glob pattern in the selected sources.
    append_detector_modules: bool
        Combine multi-module detector data in a single data source (sources for
        individual modules are removed). The last section of the source name is
        replaces with 'APPEND', example:
            'SPB_DET_AGIPD1M-1/DET/#CH0:xtdf' -> 'SPB_DET_AGIPD1M-1/DET/APPEND'

        Supported detectors: AGIPD, DSSC, LPD
    dummy_timestamps: bool
        Whether to add mock timestamps if the metadata lacks them.
    use_infiniband: bool
        Use infiniband interface if available (if port specifies a TCP port)
    sock: str
        socket type - supported: REP, PUB, PUSH (default REP).
    """
    if osp.isdir(path):
        data = RunDirectory(path)
    else:
        data = H5File(path)

    data = data.select(source_glob, key_glob)

    if isinstance(port, int) or port.isdigit():
        endpt = f'tcp://{find_infiniband_ip() if use_infiniband else "*"}:{port}'
    else:
        endpt = port
    sender = Sender(endpt, sock=sock, dummy_timestamps=dummy_timestamps)
    print(f'Streamer started on: {sender.endpoint}')
    for tid, data in _iter_trains(data, merge_detector=append_detector_modules):
        sender.send(data)

    # The karabo-bridge code sets linger to 0 so that it doesn't get stuck if
    # the client goes away. But this would also mean that we close the socket
    # when the last messages have been queued but not sent. So if we've
    # successfully queued all the messages, set linger -1 (i.e. infinite) to
    # wait until ZMQ has finished transferring them before the socket is closed.
    sender.server_socket.close(linger=-1)


def main(argv=None):
    ap = ArgumentParser(prog="karabo-bridge-serve-files")
    ap.add_argument("path", help="Path of a file or run directory to serve")
    ap.add_argument("port", help="TCP port or ZMQ endpoint to send data on")
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
        "--use-infiniband", help="Use infiniband interface if available "
                                 "(if a TCP port is specified)",
        action='store_true'
    )
    ap.add_argument(
        "-z", "--socket-type", help="ZeroMQ socket type",
        choices=['PUB', 'PUSH', 'REP'], default='REP'
    )
    args = ap.parse_args(argv)

    try:
        serve_files(
            args.path, args.port, source_glob=args.source, key_glob=args.key,
            append_detector_modules=args.append_detector_modules,
            dummy_timestamps=args.dummy_timestamps,
            use_infiniband=args.use_infiniband, sock=args.socket_type
        )
    except KeyboardInterrupt:
        pass
    print('\nStopped.')
