
import sys
from argparse import ArgumentParser
from multiprocessing import get_context
from multiprocessing.pool import ThreadPool
from os import cpu_count
from pathlib import Path
from time import monotonic

from extra_data import FileAccess
from extra_data.locality import get_locality


def check_access(path_states, path):
    start = monotonic()
    locality = get_locality(path)

    if locality == 'UNAVAILABLE':
        path_states[path] = 'dCache reports unavailable'
        return
    elif locality == 'NEARLINE':
        path_states[path] = 'dCache reports tape only'
        return
    elif locality == 'NOT_ON_DCACHE':
        prefix = 'file on GPFS but '
    else:
        prefix = 'dCache reports on disk but '

    path_states[path] = f'{prefix}stat hangs'

    path.stat()
    path_states[path] = f'{prefix}read attempt hangs'

    try:
        fa = FileAccess(path)
    except Exception as e:
        path_states[path] = f'{prefix}read attempt raises: {str(e)}'
    else:
        path_states[path] = monotonic() - start


def main(argv=None):
    ap = ArgumentParser(description='Check whether European XFEL data files '
                                    'are readable')

    ap.add_argument(
        'input', metavar='INPUT', nargs='+', type=Path,
        help='folder of input data to check')

    ap.add_argument(
        '--recursive', '-r', action='store_true',
        help='whether to search directories passed as input recursively for '
             'those containing HDF5 files')

    ap.add_argument(
        '--timeout', action='store', type=float, default=3.0,
        metavar='SECS', help='timeout for access checks, 5s by default')

    args = ap.parse_args(argv)

    # Ensure the spawn method is used to reliably terminate with no
    # zombie processes in D state lying around.
    mp_ctx = get_context('spawn')

    # Record last state for each path in a shared dictionary.
    path_states = mp_ctx.Manager().dict()

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

    def monitor_access_check(path):
        path_states[path] = 'any access hangs'
        p = mp_ctx.Process(target=check_access, args=(path_states, path,),
                           daemon=True)
        p.start()
        p.join(timeout=args.timeout)

        if p.exitcode is None:
            p.kill()
            return 'T'

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

    print('')

    # Collect access time measurement for successful checks
    times = [timing for timing in path_states.values()
             if isinstance(timing, float)]
    print(f'average access time: {(sum(times) / len(times)):03g}s, '
          f'max access time: {max(times):03g}s')

    # Collect all anomalous paths and print.
    anomalous_states = {str(path): state for path, state in path_states.items()
                        if isinstance(state, str)}

    if not anomalous_states:
        print('all files readable')
    else:
        path_col = max([len(path_str) for path_str in anomalous_states]) + 3

        for path_str, state in anomalous_states.items():
            print(path_str.ljust(path_col), state)


if __name__ == '__main__':
    sys.exit(main())