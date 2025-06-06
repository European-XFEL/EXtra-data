"""Summarise XFEL data in files or folders
"""
import argparse
from collections import defaultdict
import os
import os.path as osp
import re
import sys

from ..read_machinery import FilenameInfo
from ..reader import H5File, RunDirectory


def describe_file(path, details_for_sources=(), with_aggregators=False):
    """Describe a single HDF5 data file"""
    basename = os.path.basename(path)
    print(basename, ": Data file")

    h5file = H5File(path)
    h5file.info(details_for_sources, with_aggregators)


def summarise_file(path):
    basename = os.path.basename(path)
    print(basename, ": Data file")

    f = H5File(path)
    print(f"  {len(f.train_ids)} trains, {len(f.all_sources)} sources")


def describe_run(path, details_for_sources=(), with_aggregators=False):
    basename = os.path.basename(path)
    print(basename, ": Run directory")
    print()

    run = RunDirectory(path)
    run.info(details_for_sources, with_aggregators)


def summarise_run(path, indent=''):
    basename = os.path.basename(path)

    # Accessing all the files in a run can be slow. To get the number of trains,
    # pick one set of segments (time slices of data from the same source).
    # This relies on each set of segments recording the same number of trains.
    segment_sequences = defaultdict(list)
    n_detector = n_other = 0
    for f in sorted(os.listdir(path)):
        m = re.match(r'(.+)-S\d+\.h5', osp.basename(f))
        if m:
            segment_sequences[m.group(1)].append(f)
            if FilenameInfo(f).is_detector:
                n_detector += 1
            else:
                n_other += 1

    if len(segment_sequences) < 1:
        raise ValueError("No data files recognised in %s" % path)

    # Take the shortest group of segments to make reading quicker
    first_group = sorted(segment_sequences.values(), key=len)[0]
    train_ids = set()
    for f in first_group:
        train_ids.update(H5File(osp.join(path, f)).train_ids)

    print("{}{} : Run of {:>4} trains, with {:>3} detector files and {:>3} others".format(
        indent, basename, len(train_ids), n_detector, n_other
    ))

def main(argv=None):
    ap = argparse.ArgumentParser(
        prog='lsxfel', description="Summarise XFEL data in files or folders"
    )
    ap.add_argument('paths', nargs='*', help="Files/folders to look at")
    ap.add_argument('--detail', '-d', action='append', default=[],
        help="Show details on keys & data for specified sources. "
             "This can slow down lsxfel considerably. "
             "Wildcard patterns like '*XGM/*:output' are allowed, or partial "
             "matches for the source name like 'XGM' with no wildcards. "
             "Can be used more than once to include several patterns. "
             "Only used when inspecting a single run or file."
    )
    ap.add_argument('--aggregators', '-g', action='store_true',
        help="Include the aggregator each source is saved in. "
             "This can slow down lsxfel considerably. ")
    args = ap.parse_args(argv)
    paths = args.paths or [os.path.abspath(os.getcwd())]

    # If --detail doesn't contain glob wildcards, treat it as a substring
    detail_patterns = [p if re.search(r'[*?[]', p) else f'*{p}*'
                       for p in args.detail]

    if len(paths) == 1:
        path = paths[0]
        basename = os.path.basename(os.path.abspath(path.rstrip('/')))

        if os.path.isdir(path):
            contents = sorted(os.listdir(path))
            if any(f.endswith('.h5') for f in contents):
                # Run directory
                describe_run(path, detail_patterns, args.aggregators)
            elif any(re.match(r'r\d+', f) for f in contents):
                # Proposal directory, containing runs
                print(basename, ": Proposal data directory")
                print()
                for f in contents:
                    child_path = os.path.join(path, f)
                    if re.match(r'r\d+', f) and os.path.isdir(child_path):
                        summarise_run(child_path, indent='  ')
            elif osp.isdir(osp.join(path, 'raw')):
                print(basename, ": Proposal directory")
                print()
                print('{}/raw/'.format(basename))
                for f in sorted(os.listdir(osp.join(path, 'raw'))):
                    child_path = os.path.join(path, 'raw', f)
                    if re.match(r'r\d+', f) and os.path.isdir(child_path):
                        summarise_run(child_path, indent='  ')
            else:
                print(basename, ": Unrecognised directory")
        elif os.path.isfile(path):
            if path.endswith('.h5'):
                describe_file(path, detail_patterns, args.aggregators)
            else:
                print(basename, ": Unrecognised file")
                return 2
        else:
            print(path, ': File/folder not found')
            return 2
    else:
        exit_code = 0
        for path in paths:
            basename = os.path.basename(path)

            if os.path.isdir(path):
                contents = os.listdir(path)
                if any(f.endswith('.h5') for f in contents):
                    # Run directory
                    summarise_run(path)
                elif any(re.match(r'r\d+', f) for f in contents):
                    # Proposal directory, containing runs
                    print(basename, ": Proposal directory")
                    print()
                    for f in contents:
                        child_path = os.path.join(path, f)
                        if re.match(r'r\d+', f) and os.path.isdir(child_path):
                            summarise_run(child_path, indent='  ')
                else:
                    print(basename, ": Unrecognised directory")
                    exit_code = 2
            elif os.path.isfile(path):
                if path.endswith('.h5'):
                    summarise_file(path)
                else:
                    print(basename, ": Unrecognised file")
                    exit_code = 2
            else:
                print(path, ': File/folder not found')
                exit_code = 2

        return exit_code


if __name__ == '__main__':
    sys.exit(main())
