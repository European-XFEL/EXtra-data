
import sys
from argparse import ArgumentParser
import subprocess
from multiprocessing.pool import ThreadPool
from os import cpu_count
from pathlib import Path
from time import monotonic

from extra_data import FileAccess
from extra_data.locality import get_locality


def check_access(path):
    print('any access hangs', flush=True)

    start = monotonic()
    locality = get_locality(path)

    if locality == 'UNAVAILABLE':
        print('dCache reports unavailable', flush=True)
        return 0
    elif locality == 'NEARLINE':
        print('dCache reports tape only', flush=True)
        return 0
    elif locality == 'NOT_ON_DCACHE':
        prefix = 'file on GPFS but '
    else:
        prefix = 'dCache reports on disk but '

    print(f'{prefix}stat hangs', flush=True)

    path.stat()
    print(f'{prefix}read attempt hangs', flush=True)

    try:
        fa = FileAccess(path)
    except Exception as e:
        print(f'{prefix}read attempt raises: {str(e)}')
    else:
        print(f'readable_{monotonic() - start}')

    return 0


check_access_runtime = '''
import sys
from pathlib import Path
from extra_data.cli.check_readable import check_access
sys.exit(check_access(Path('{}')))
'''


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

    def monitor_access_check(path):
        try:
            p = subprocess.run(
                [sys.executable, '-c', check_access_runtime.format(path)],
                timeout=args.timeout, capture_output=True)
        except subprocess.TimeoutExpired as e:
            path_states[path] = e.stdout.decode().split('\n')[-2]
            return 'T'
        else:
            path_states[path] = p.stdout.decode().split('\n')[-2]
            return '.'

    with ThreadPool(processes=min(10, cpu_count() // 4)) as pool:
        for res in pool.imap_unordered(monitor_access_check, paths):
            print(res, end='', flush=True)

    '''
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=5) as pool:
        for _ in pool.map(monitor_access_check, paths):
            pass
    '''

    '''
    for path in paths:
        print(monitor_access_check(path), end='', flush=True)
    '''

    print('')

    # Collect access time measurement for successful checks
    times = [float(state[9:]) for state in path_states.values()
             if state.startswith('readable')]
    print(f'average access time: {(sum(times) / len(times)):03g}s, '
          f'max access time: {max(times):03g}s')

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


if __name__ == '__main__':
    sys.exit(main())