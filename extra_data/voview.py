"""Create & check 'virtual overview' files

These use virtual datasets to present the data from a run as a single file.
"""

import os
import os.path as osp
import re
import sys
from tempfile import TemporaryDirectory

import h5py

from .file_access import FileAccess
from .writer import VirtualFileWriter

DATA_ROOT_DIR = "/gpfs/exfel/exp/"
# Version number for virtual overview format - increment if we need to stop old
# versions of EXtra-data from reading files made by newer versions.
VOVIEW_VERSION = 1


class VirtualOverviewFileWriter(VirtualFileWriter):
    def record_source_files(self):
        grp = self.file.create_group('.source_files')
        names, sizes = [], []
        for fa in self.data.files:
            st = fa.metadata_fstat or os.stat(fa.filename)
            names.append(osp.basename(fa.filename).encode('ascii'))
            sizes.append(st.st_size)

        grp.create_dataset(
            'names', data=names, dtype=h5py.special_dtype(vlen=bytes)
        )
        grp.create_dataset('sizes', data=sizes, dtype='u8')

    def write(self):
        self.record_source_files()
        self.file.attrs['virtual_overview_version'] = VOVIEW_VERSION
        super().write()


def check_sources(overview_file: h5py.File, run_dir):
    g = overview_file['.source_files']
    if not (g['names'].shape == g['sizes'].shape):
        return False  # Basic check that things make sense

    files_now = {f for f in os.listdir(run_dir)
                 if f.endswith('.h5') and (f.lower() != 'overview.h5')}
    files_stored = [p.decode('ascii') for p in g['names'][:]]
    if files_now != set(files_stored):
        return False

    for name, size in zip(files_stored, g['sizes']):
        st = os.stat(osp.join(run_dir, name))
        if st.st_size != size:
            return False

    return True


def voview_paths_for_run(directory):
    paths = [osp.join(directory, 'overview.h5')]
    # After resolving symlinks, data on Maxwell is stored in either
    # GPFS, e.g. /gpfs/exfel/d/proc/SCS/201901/p002212  or
    # dCache, e.g. /pnfs/xfel.eu/exfel/archive/XFEL/raw/SCS/201901/p002212
    # On the online cluster the resolved path stay:
    #   /gpfs/exfel/exp/inst/cycle/prop/(raw|proc)/run
    maxwell_match = re.match(
        #     raw/proc  instr  cycle prop   run
        r'.+/(raw|proc)/(\w+)/(\w+)/(p\d+)/(r\d+)/?$',
        osp.realpath(directory)
    )
    online_match = re.match(
        #     instr cycle prop   raw/proc   run
        r'^.+/(\w+)/(\w+)/(p\d+)/(raw|proc)/(r\d+)/?$',
        osp.realpath(directory)
    )

    if maxwell_match:
        raw_proc, instr, cycle, prop, run_nr = maxwell_match.groups()
    elif online_match:
        instr, cycle, prop, raw_proc, run_nr = online_match.groups()
    else:
        return paths

    fname = f'{raw_proc.upper()}-{run_nr.upper()}-OVERVIEW.h5'
    prop_usr = osp.join(
        DATA_ROOT_DIR, instr, cycle, prop, 'usr'
    )
    if osp.isdir(prop_usr):
        paths.append(
            osp.join(prop_usr, '.extra_data', fname)
        )
    return paths

def find_file_read(run_dir):
    for candidate in voview_paths_for_run(run_dir):
        if osp.isfile(candidate):
            return candidate

def find_file_valid(run_dir):
    for candidate in voview_paths_for_run(run_dir):
        if h5py.is_hdf5(candidate):
            file_acc = FileAccess(candidate)
            version = file_acc.file.attrs.get('virtual_overview_version', 0)
            if version <= VOVIEW_VERSION and check_sources(file_acc.file, run_dir):
                return file_acc

def find_file_write(run_dir):
    for candidate in voview_paths_for_run(run_dir):
        try:
            os.makedirs(osp.dirname(candidate), exist_ok=True)
            candidate_tmp = candidate + '.check'
            with open(candidate_tmp, 'wb'):
                pass
            os.unlink(candidate_tmp)
            return candidate
        except PermissionError:
            pass

    raise PermissionError


def write_atomic(path, data):
    """Write a virtual overview file, then rename it to the final path

    This aims to avoid exposing a partially written file where EXtra-data might
    try to read it.
    """
    dirname, basename = osp.split(path)
    with TemporaryDirectory(prefix=".create-voview-", dir=dirname) as td:
        tmp_filename = osp.join(td, basename)
        try:
            with VirtualOverviewFileWriter(tmp_filename, data) as vofw:
                vofw.write()
            os.replace(tmp_filename, path)
        except:
            os.unlink(tmp_filename)
            raise


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('run_dir')
    ap.add_argument('--overview-file')
    args = ap.parse_args(argv)

    if args.check:
        file_path = args.overview_file or find_file_read(args.run_dir)
        print(f"Checking {file_path} ...")
        with h5py.File(file_path, 'r') as f:
            ok = check_sources(f, args.run_dir)
        if ok:
            print("Source files match, overview file can be used")
        else:
            print("Source files don't match, overview file outdated")
            return 1
    else:
        from . import RunDirectory
        file_path = args.overview_file or find_file_write(args.run_dir)
        print("Opening", args.run_dir)
        run = RunDirectory(args.run_dir, _use_voview=False)
        print(f"Creating {file_path} from {len(run.files)} files...")
        write_atomic(file_path, run)

if __name__ == '__main__':
    sys.exit(main())
