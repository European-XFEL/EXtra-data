
import subprocess
import sys
from argparse import ArgumentParser
from multiprocessing.pool import ThreadPool
from os import cpu_count
from pathlib import Path
from time import monotonic

from extra_data import FileAccess
from extra_data.locality import get_locality


def check_access(path):
    print('any access hangs', flush=True)

    start = monotonic()

    try:
        _, locality = get_locality(path)
    except Exception as e:
        print(f'locality check raises {e.__class__.__name__}')
        return 1

    if locality == 'UNAVAILABLE':
        print('dCache reports unavailable')
        return 0
    elif locality == 'NEARLINE':
        print('dCache reports tape only')
        return 0
    elif locality == 'NOT_ON_DCACHE':
        prefix = 'file on GPFS but '
    else:
        prefix = 'dCache reports on disk but '

    print(f'{prefix}stat hangs', flush=True)

    try:
        path.stat()
    except Exception as e:
        print(f'{prefix}stat raises {e.__class__.__name__}')
        return 1
    else:
        print(f'{prefix}read attempt hangs', flush=True)

    try:
        fa = FileAccess(path)
    except Exception as e:
        print(f'{prefix}read attempt raises {e.__class__.__name__}')
    else:
        fa.file.close()
        print(f'readable_{monotonic() - start}')

    return 0


def main(argv=None):
    ap = ArgumentParser(description='Check whether European XFEL data files '
                                    'are readable')

    ap.add_argument(
        'input', metavar='INPUT', nargs='+', type=Path,
        help='folder of input data to check')

    ap.add_argument(
        '--all', '-a', action='store_true',
        help='whether to show the result for all files rather than only '
             'unreadable ones')

    ap.add_argument(
        '--recursive', '-r', action='store_true',
        help='whether to search directories passed as input recursively for '
             'those containing HDF5 files')

    ap.add_argument(
        '--timeout', action='store', type=float, default=5.0,
        metavar='SECS', help='timeout for access checks, 5s by default')

    args = ap.parse_args(argv)

    # Collect input files to check.
    paths = []

    for inp in args.input:
        if inp.is_dir():
            paths.extend(inp.glob('**/*.h5' if args.recursive else '*.h5'))
        elif inp.is_file():
            paths.append(inp)

    if not paths:
        print('No HDF5 files to check')
        return 0

    # Map of the result for each path.
    path_states = {}

    # Bootstrap code to run access check
    check_access_runtime = '''
import sys
from pathlib import Path
from extra_data.cli.check_readable import check_access
sys.exit(check_access(Path({!r})))
    '''

    def monitor_access_check(path):
        try:
            # OMP_NUM_THREADS=1 minimizes creation of thread pools when
            # numpy is imported in the subprocess. As the subprocesses
            # often live less than 1s for successful checks, this can
            # cause significant load on cluster-sized nodes.
            p = subprocess.run(
                [sys.executable, '-c', check_access_runtime.format(str(path))],
                timeout=args.timeout, capture_output=True,
                env={'OMP_NUM_THREADS': '1'})
        except subprocess.TimeoutExpired as e:
            path_states[path] = e.stdout.decode().splitlines()[-1]
            return 'T'
        else:
            path_states[path] = p.stdout.decode().splitlines()[-1]

            if path_states[path].startswith('readable'):
                return '.'  # Successful check
            elif p.returncode != 0:
                return 'E'  # Exception raised
            else:
                return 'X'  # Graceful unavailable

    with ThreadPool(processes=min(32, cpu_count() // 4)) as pool:
        for res in pool.imap_unordered(monitor_access_check, paths):
            print(res, end='', flush=True)

    print('')

    # Collect access time measurement for successful checks
    times = [float(state[9:]) for state in path_states.values()
             if state.startswith('readable')]

    if not times:
        print('no files are readable')
    else:
        print(f'average access time: {(sum(times) / len(times)):.3g}s, '
              f'max access time: {max(times):.3g}s')

    # Collect all paths to be shown.
    shown_states = {str(path): state.partition('_')[0] for path, state
                    in path_states.items()
                    if args.all or not state.startswith('readable')}

    if not shown_states:
        print('all files readable')
    else:
        path_col = max([len(path_str) for path_str in shown_states]) + 3

        for path_str in sorted(shown_states):
            print(path_str.ljust(path_col), shown_states[path_str])

    return 0 if len(times) == len(paths) else 1


if __name__ == '__main__':
    sys.exit(main())