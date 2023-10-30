from argparse import ArgumentParser
import sys

from .. import open_run

IMPORT_FAILED_MSG = """\
{}

karabo-bridge-serve-run requires additional dependencies:
    pip install karabo-bridge psutil
"""

def main(argv=None):
    ap = ArgumentParser(prog="karabo-bridge-serve-run")
    ap.add_argument("proposal", help="Proposal number")
    ap.add_argument("run", help="Run number")
    ap.add_argument(
        "--port", default="0", help="TCP port or ZMQ endpoint to send data on. "
                                    "Selects a random TCP port by default.")
    ap.add_argument(
        "--include", help="Select matching sources (and optionally keys) to "
                          "include in streamed data",
        action='append'
    )
    ap.add_argument(
        "--allow-partial", help="Send trains where some sources are missing",
        action='store_true'
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
        from ..export import serve_data
    except ImportError as e:
        sys.exit(IMPORT_FAILED_MSG.format(e))

    run = open_run(args.proposal, args.run, data='all')

    if not args.include:
        print("Available sources:")
        for s in sorted(run.all_sources):
            print(f"  {s}")
        sys.exit("Please select at least one source with --include")

    include = []
    for pat in args.include:
        if '[' in pat:
            if not pat.endswith(']'):
                sys.exit(f"Missing final ] in {pat!r}")
            src_pat, key_pat = pat[:-1].split('[', 1)
            include.append((src_pat, key_pat))
        else:
            # Source pattern only
            include.append(pat)

    if args.allow_partial:
        sel = run.select(include, require_any=True)
    else:
        sel = run.select(include, require_all=True)

    try:
        serve_data(
            sel, args.port,
            append_detector_modules=args.append_detector_modules,
            dummy_timestamps=args.dummy_timestamps,
            use_infiniband=args.use_infiniband, sock=args.socket_type
        )
    except KeyboardInterrupt:
        print('\nStopped.')

if __name__ == '__main__':
    main()
