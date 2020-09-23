from argparse import ArgumentParser
from multiprocessing import Pool
from functools import partial
import numpy as np
import os
import os.path as osp
from shutil import get_terminal_size
from signal import signal, SIGINT, SIG_IGN
import sys

from .reader import H5File, FileAccess
from .run_files_map import RunFilesMap


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


class FileValidator:
    def __init__(self, file: FileAccess):
        self.file = file
        self.filename = file.filename
        self.problems = []

    def validate(self):
        problems = self.run_checks()
        if problems:
            raise ValidationError(problems)

    def run_checks(self):
        self.problems = []
        self.check_indices()
        self.check_trainids()
        return self.problems

    def record(self, msg, **kwargs):
        self.problems.append(dict(msg=msg, file=self.filename, **kwargs))

    def check_trainids(self):
        ds_path = 'INDEX/trainId'
        train_ids = self.file.file[ds_path][:]

        if (train_ids == 0).any():
            first0 = train_ids.tolist().index(0)
            if not (train_ids[first0:] == 0).all():
                self.record(
                    'Zeroes in trainId index before last train ID', dataset=ds_path
                )
            nonzero_tids = train_ids[train_ids != 0]
        else:
            nonzero_tids = train_ids

        if len(nonzero_tids) > 1:
            non_incr = (nonzero_tids[1:] <= nonzero_tids[:-1]).nonzero()[0]
            if non_incr.size > 0:
                pos = non_incr[0]
                self.record(
                    'Train IDs are not strictly increasing, e.g. at {} ({} >= {})'.format(
                        pos, nonzero_tids[pos], nonzero_tids[pos + 1]
                    ),
                    dataset=ds_path,
                )

    def _get_index(self, source, group):
        """returns first and count dataset for specified source.
        
        This is slightly different to the same method in FileAccess as it does
        cut the dataset up to the trainId's dataset length.
        """
        ix_group = self.file.file['/INDEX/{}/{}'.format(source, group)]
        firsts = ix_group['first'][:]
        if 'count' in ix_group:
            counts = ix_group['count'][:]
        else:
            status = ix_group['status'][:]
            counts = np.uint64((ix_group['last'][:] - firsts + 1) * status)
        return firsts, counts

    def check_indices(self):
        for src in self.file.instrument_sources:
            src_groups = set()
            for key in self.file.get_keys(src):
                ds_path = 'INSTRUMENT/{}/{}'.format(src, key.replace('.', '/'))
                group = key.split('.', 1)[0]
                src_groups.add((src, group))
                first, count = self.file.get_index(src, group)
                data_dim0 = self.file.file[ds_path].shape[0]
                if np.any((first + count) > data_dim0):
                    max_end = (first + count).max()
                    self.record(
                        'Index referring to data ({}) outside dataset ({})'.format(
                            max_end, data_dim0
                        ),
                        dataset=ds_path,
                    )

            for src, group in src_groups:
                record = partial(self.record, dataset='INDEX/{}/{}'.format(src, group))
                first, count = self._get_index(src, group)

                if (first.ndim != 1) or (count.ndim != 1):
                    record(
                        "Index first / count are not 1D",
                        first_shape=first.shape,
                        count_shape=count.shape,
                    )
                    continue

                if first.shape != count.shape:
                    record(
                        "Index first & count have different number of entries",
                        first_shape=first.shape,
                        count_shape=count.shape,
                    )
                    continue

                if first.shape != self.file.train_ids.shape:
                    record(
                        "Index has wrong number of entries",
                        index_shape=first.shape,
                        trainids_shape=self.file.train_ids.shape,
                    )

                check_index_contiguous(first, count, record)


def check_index_contiguous(firsts, counts, record):
    if firsts.size == 0:
        return  # no data in this dataset

    if firsts[0] != 0:
        record("Index doesn't start at 0")

    gaps = firsts[1:].astype(np.int64) - (firsts + counts)[:-1]

    gap_ixs = (gaps > 0).nonzero()[0]
    if gap_ixs.size > 0:
        pos = gap_ixs[0]
        record("Gaps ({}) in index, e.g. at {} ({} + {} < {})".format(
            gap_ixs.size, pos, firsts[pos], counts[pos], firsts[pos+1]
        ))

    overlap_ixs = (gaps < 0).nonzero()[0]
    if overlap_ixs.size > 0:
        pos = overlap_ixs[0]
        record("Overlaps ({}) in index, e.g. at {} ({} + {} > {})".format(
            overlap_ixs.size, pos, firsts[pos], counts[pos], firsts[pos + 1]
        ))


def progress_bar(done, total, suffix=' '):
    line = f'Progress: {done}/{total}{suffix}[{{}}]'
    length = min(get_terminal_size().columns - len(line), 50)
    filled = int(length * done // total)
    bar = '#' * filled + ' ' * (length - filled)
    return line.format(bar)


def _check_file(args):
    runpath, filename = args
    filepath = osp.join(runpath, filename)
    problems = []
    try:
        fa = FileAccess(filepath)
    except Exception as e:
        problems.append(
            dict(msg="Could not open file", file=filepath, error=e)
        )
        return filename, None, problems
    else:
        fv = FileValidator(fa)
        problems.extend(fv.run_checks())
        fa.close()
    return filename, fa, problems


class RunValidator:
    def __init__(self, run_dir: str, term_progress=False):
        self.run_dir = run_dir
        self.term_progress = term_progress
        self.filenames = [f for f in os.listdir(run_dir) if f.endswith('.h5')]
        self.file_accesses = []
        self.problems = []

    def validate(self):
        problems = self.run_checks()
        if problems:
            raise ValidationError(problems)

    def run_checks(self):
        self.problems = []
        self.check_files()
        self.check_files_map()
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
            iterator = pool.imap_unordered(_check_file, filepaths)
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

    def check_files_map(self):
        # Outdated cache entries we can detect with the file's stat() are not a
        # problem. Loading the cache file will discard those automatically.
        cache = RunFilesMap(self.run_dir)
        for f_access in self.file_accesses:
            f_cache = cache.get(f_access.filename)
            if f_cache is None:
                continue

            if (
                    f_cache['control_sources'] != f_access.control_sources
                 or f_cache['instrument_sources'] != f_access.instrument_sources
                 or not np.array_equal(f_cache['train_ids'], f_access.train_ids)
            ):
                self.problems.append(dict(
                    msg="Incorrect data map cache entry",
                    cache_file=cache.cache_file,
                    data_file=f_access.filename,
                ))

            f_access.close()

def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]

    ap = ArgumentParser(prog='extra-data-validate')
    ap.add_argument('path', help="HDF5 file or run directory of HDF5 files.")
    args = ap.parse_args(argv)

    path = args.path
    if os.path.isdir(path):
        print("Checking run directory:", path)
        print()
        validator = RunValidator(path, term_progress=True)
    else:
        print("Checking file:", path)
        validator = FileValidator(H5File(path).files[0])

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
