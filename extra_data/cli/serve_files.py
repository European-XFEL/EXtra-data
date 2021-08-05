from argparse import ArgumentParser
import sys

IMPORT_FAILED_MSG = """\
{}

karabo-bridge-serve-files requires additional dependencies:
    pip install karabo-bridge psutil
"""

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
        from ..export import serve_files
    except ImportError as e:
        sys.exit(IMPORT_FAILED_MSG.format(e))

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

if __name__ == '__main__':
    main()
