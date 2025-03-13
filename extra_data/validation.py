from argparse import ArgumentParser, Action
from multiprocessing import Pool
from functools import partial
import numpy as np
import os
import os.path as osp
from signal import signal, SIGINT, SIG_IGN
import sys

from .reader import H5File, FileAccess
from .run_files_map import RunFilesMap
from .utils import progress_bar

class ValidationError(Exception):
    def __init__(self, problems):
        self.problems = problems

    def __str__(self):
        lines = []
        for prob in self.problems:
            lines.extend(['', prob['msg']])
            for k, v in sorted(prob.items()):
                if k != 'msg':
                    lines.append("  {}: {}".format(k, v))

        return '\n'.join(lines)


def problem(msg, **kwargs):
    return dict(msg=msg, **kwargs)


class FileValidator:
    def __init__(self, file: FileAccess, skip_checks=()):
        self.file = file
        self.filename = file.filename
        self.problems = []
        self.skip_checks = set(skip_checks)

    check_funcs = []

    def validate(self):
        problems = self.run_checks()
        if problems:
            raise ValidationError(problems)

    def add_filename(self, prob: dict):
        prob['file'] = self.filename
        return prob

    def run_checks(self):
        self.problems = []
        for func in self.check_funcs:
            if func.__name__ in self.skip_checks:
                continue
            self.problems += [self.add_filename(p) for p in func(self.file)]

        return self.problems


def file_check(f):
    FileValidator.check_funcs.append(f)
    return f


@file_check
def train_ids_nonzero(file):
    ds_path = 'INDEX/trainId'
    train_ids = file.file[ds_path][:]

    if (train_ids == 0).any():
        first0 = train_ids.tolist().index(0)
        if not (train_ids[first0:] == 0).all():
            yield problem(
                'Zeroes in trainId index before last train ID', dataset=ds_path
            )


@file_check
def train_ids_order(file):
    ds_path = 'INDEX/trainId'
    train_ids = file.file[ds_path][:]
    nonzero_tids = train_ids[train_ids != 0]

    if len(nonzero_tids) > 1:
        non_incr = (nonzero_tids[1:] <= nonzero_tids[:-1]).nonzero()[0]
        if non_incr.size > 0:
            pos = non_incr[0]
            yield problem(
                'Train IDs are not strictly increasing, e.g. at {} ({} >= {})'.format(
                    pos, nonzero_tids[pos], nonzero_tids[pos + 1]
                ),
                dataset=ds_path,
            )


@file_check
def index_control(file):
    for src in file.control_sources:
        first, count = file.get_index(src, '')
        for key in file.get_keys(src):
            ds_path = f"CONTROL/{src}/{key.replace('.', '/')}"
            data_dim0 = file.file[ds_path].shape[0]
            if np.any((first + count) > data_dim0):
                max_end = (first + count).max()
                yield problem(
                    'Index referring to data ({}) outside dataset ({})'.format(
                        max_end, data_dim0
                    ),
                    dataset=ds_path,
                )
                break  # Recording every key separately can make a *lot* of errors

        yield from _check_index(file, f'INDEX/{src}')

@file_check
def index_instrument(file):
    for src in file.instrument_sources:
        src_groups = set()
        for key in file.get_keys(src):
            ds_path = 'INSTRUMENT/{}/{}'.format(src, key.replace('.', '/'))
            group = key.split('.', 1)[0]
            src_groups.add((src, group))
            first, count = file.get_index(src, group)
            data_dim0 = file.file[ds_path].shape[0]
            if np.any((first + count) > data_dim0):
                max_end = (first + count).max()
                yield problem(
                    'Index referring to data ({}) outside dataset ({})'.format(
                        max_end, data_dim0
                    ),
                    dataset=ds_path,
                )

        for src, group in src_groups:
            yield from _check_index(file, f'INDEX/{src}/{group}')

def _get_index(file, path):
    """returns first and count dataset for specified source.

    This is slightly different to the same method in FileAccess as it does
    cut the dataset up to the trainId's dataset length.
    """
    ix_group = file.file[path]
    firsts = ix_group['first'][:]
    if 'count' in ix_group:
        counts = ix_group['count'][:]
    else:
        status = ix_group['status'][:]
        counts = np.uint64((ix_group['last'][:] - firsts + 1) * status)
    return firsts, counts

def _check_index(file, path):
    ds_problem = partial(problem, dataset=path)
    first, count = _get_index(file, path)

    if (first.ndim != 1) or (count.ndim != 1):
        yield ds_problem(
            "Index first / count are not 1D",
            first_shape=first.shape,
            count_shape=count.shape,
        )
        return

    if first.shape != count.shape:
        yield ds_problem(
            "Index first & count have different number of entries",
            first_shape=first.shape,
            count_shape=count.shape,
        )
        return

    if first.shape != file.train_ids.shape:
        yield ds_problem(
            "Index has wrong number of entries",
            index_shape=first.shape,
            trainids_shape=file.train_ids.shape,
        )

    yield from check_index_contiguous(first, count, ds_problem)


def check_index_contiguous(firsts, counts, ds_problem):
    if firsts.size == 0:
        return  # no data in this dataset

    if firsts[0] != 0:
        yield ds_problem("Index doesn't start at 0")

    gaps = firsts[1:].astype(np.int64) - (firsts + counts)[:-1]

    gap_ixs = (gaps > 0).nonzero()[0]
    if gap_ixs.size > 0:
        pos = gap_ixs[0]
        yield ds_problem("Gaps ({}) in index, e.g. at {} ({} + {} < {})".format(
            gap_ixs.size, pos, firsts[pos], counts[pos], firsts[pos+1]
        ))

    overlap_ixs = (gaps < 0).nonzero()[0]
    if overlap_ixs.size > 0:
        pos = overlap_ixs[0]
        yield ds_problem("Overlaps ({}) in index, e.g. at {} ({} + {} > {})".format(
            overlap_ixs.size, pos, firsts[pos], counts[pos], firsts[pos + 1]
        ))


@file_check
def control_timestamps_order(file):
    """Check that CONTROL value's timestamps are monotonically increasing.
    """
    for source in file.control_sources:
        for key in file.get_keys(source):
            if not key.endswith('.timestamp'):
                continue

            ds_path = f'CONTROL/{source}/{key.replace(".", "/")}'
            ts = file.file[ds_path][:]

            if (ts == 0).any():
                first0 = np.where(ts == 0)[0][0]
                if not (ts[first0:] == 0).all():
                    yield problem(
                        'Zeroes in Timestamp before last train ID',
                        dataset=ds_path
                    )
                nonzero_ts = ts[:first0]
            else:
                nonzero_ts = ts

            non_incr = (nonzero_ts[1:] < nonzero_ts[:-1]).nonzero()[0]
            if non_incr.size > 0:
                pos = non_incr[0]
                yield problem(
                    f'Timestamp is decreasing, e.g. at '
                    f'{pos + 1} ({ts[pos + 1]} < {ts[pos]})',
                    dataset=ds_path,
                )


def _open_file(filepath):
    try:
        fa = FileAccess(filepath)
    except Exception as e:
        try:
            with open(filepath, "rb") as f:
                f.read(16)
        except OSError as e2:
            # Filesystem issue, e.g. dCache node down. HDF5 errors can be
            # confusing, so record the OS error instead.
            pb = dict(msg="Could not access file", file=filepath, error=e2)
        else:
            # HDF5 file corrupted or missing expected information
            pb = dict(msg="Could not open HDF5 file", file=filepath, error=e)
        return None, [pb]
    else:
        return fa, []


class RunValidator:
    def __init__(self, run_dir: str, term_progress=False, skip_checks=()):
        self.run_dir = run_dir
        self.term_progress = term_progress
        self.filenames = [f for f in os.listdir(run_dir) if f.endswith('.h5')]
        self.file_accesses = []
        self.problems = []
        self.skip_checks = set(skip_checks)

    check_funcs = []

    def validate(self):
        problems = self.run_checks()
        if problems:
            raise ValidationError(problems)

    def run_checks(self):
        self.problems = []
        # check_files populates file_accesses as well as running FileValidator
        self.check_files()
        for func in self.check_funcs:
            if func.__name__ in self.skip_checks:
                continue
            self.problems += func(self.run_dir, self.file_accesses)

        return self.problems

    def progress(self, done, total, nproblems, badfiles):
        """Show progress information"""
        if not self.term_progress:
            return

        lines = progress_bar(done, total)
        lines += f'\n{nproblems} problems'
        if badfiles:
            lines += f' in {len(badfiles)} files (last: {badfiles[-1]})'
        if sys.stderr.isatty():
            # "\x1b[2K": delete whole line, "\x1b[1A": move up cursor
            print('\x1b[2K\x1b[1A\x1b[2K', end='\r',file=sys.stderr)
            print(lines, end='', file=sys.stderr)
        else:
            print(lines, file=sys.stderr)

    def _check_file(self, args):
        runpath, filename = args
        filepath = osp.join(runpath, filename)
        fa, problems = _open_file(filepath)
        if fa is not None:
            fv = FileValidator(fa, skip_checks=self.skip_checks)
            problems.extend(fv.run_checks())
            fa.close()
        return filename, fa, problems

    def check_files(self):
        self.file_accesses = []

        def initializer():
            # prevent child processes from receiving KeyboardInterrupt
            signal(SIGINT, SIG_IGN)

        filepaths = [(self.run_dir, fn) for fn in sorted(self.filenames)]
        nfiles = len(self.filenames)
        badfiles = []
        self.progress(0, nfiles, 0, badfiles)

        with Pool(initializer=initializer) as pool:
            iterator = pool.imap_unordered(self._check_file, filepaths)
            for done, (fname, fa, problems) in enumerate(iterator, start=1):
                if problems:
                    self.problems.extend(problems)
                    badfiles.append(fname)
                if fa is not None:
                    self.file_accesses.append(fa)
                self.progress(done, nfiles, len(self.problems), badfiles)

        if not self.file_accesses:
            self.problems.append(
                dict(msg="No usable files found", directory=self.run_dir)
            )


def run_dir_check(f):
    RunValidator.check_funcs.append(f)
    return f


@run_dir_check
def run_json_cache(run_dir, file_accesses):
    # Outdated cache entries we can detect with the file's stat() are not a
    # problem. Loading the cache file will discard those automatically.
    cache = RunFilesMap(run_dir)
    for f_access in file_accesses:
        f_cache = cache.get(f_access.filename)
        if f_cache is None:
            continue

        if (
                f_cache['control_sources'] != f_access.control_sources
             or f_cache['instrument_sources'] != f_access.instrument_sources
             or not np.array_equal(f_cache['train_ids'], f_access.train_ids)
        ):
            yield dict(
                msg="Incorrect data map cache entry",
                cache_file=cache.cache_file,
                data_file=f_access.filename,
            )

        f_access.close()


class ListAction(Action):
    def __call__(self, parser, namespace, values, option_string=None):
        print("Available checks:")
        for func in FileValidator.check_funcs + RunValidator.check_funcs:
            print(f"  {func.__name__}")
        parser.exit()


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    ap = ArgumentParser(prog='extra-data-validate')
    ap.add_argument('path', help="HDF5 file or run directory of HDF5 files.")
    ap.add_argument('-l', '--list', action=ListAction, nargs=0,
                    help="List available checks (options for --skip)")
    ap.add_argument('--skip', action='append', default=[],
                    help="Skip a named check (may be used several times)")
    args = ap.parse_args(argv)

    available_checks = {
        f.__name__ for f in FileValidator.check_funcs + RunValidator.check_funcs
    }
    bad_skips = set(args.skip) - available_checks
    if bad_skips:
        print("Unknown names passed to --skip:", ", ".join(sorted(bad_skips)))
        return 1

    path = args.path
    if os.path.isdir(path):
        print("Checking run directory:", path)
        print()
        validator = RunValidator(path, term_progress=True, skip_checks=args.skip)
    else:
        print("Checking file:", path)
        fa, problems = _open_file(path)
        if problems:
            print(str(ValidationError(problems)))
            return 1

        validator = FileValidator(fa, skip_checks=args.skip)

    try:
        validator.run_checks()
    except KeyboardInterrupt:
        print('\n^C (validation cancelled)')
    else:
        print()  # Start a new line

    if validator.problems:
        print(f"Validation failed! {len(validator.problems)} problems:")
        print(str(ValidationError(validator.problems)))
        return 1
    else:
        print("No problems found")


if __name__ == '__main__':
    sys.exit(main())
