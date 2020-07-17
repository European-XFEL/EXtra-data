"""Create & check 'virtual overview' files

These use virtual datasets to present the data from a run as a single file.
"""

import os
import os.path as osp
import re
import sys

import h5py

from .file_access import FileAccess
from .writer import VirtualFileWriter

SCRATCH_ROOT_DIR = "/gpfs/exfel/exp/"


class VirtualOverviewFileWriter(VirtualFileWriter):
    def record_source_files(self):
        grp = self.file.create_group('.source_files')
        names, mtimes, sizes = [], [], []
        for fa in self.data.files:
            st = fa.metadata_fstat or os.stat(fa.filename)
            names.append(osp.basename(fa.filename).encode('ascii'))
            mtimes.append(st.st_mtime)
            sizes.append(st.st_size)

        grp.create_dataset(
            'names', data=names, dtype=h5py.special_dtype(vlen=bytes)
        )
        grp.create_dataset('mtimes', data=mtimes, dtype='f8')
        grp.create_dataset('sizes', data=sizes, dtype='u8')

    def write(self):
        self.record_source_files()
        super().write()


def check_sources(overview_file: h5py.File, run_dir):
    g = overview_file['.source_files']
    if not (g['names'].shape == g['mtimes'].shape == g['sizes'].shape):
        return False  # Basic check that things make sense

    files_now = {f for f in os.listdir(run_dir)
                 if f.endswith('.h5') and ('overview' not in f.lower())}
    files_stored = [p.decode('ascii') for p in g['names'][:]]
    if files_now != set(files_stored):
        return False

    for name, mtime, size in zip(files_stored, g['mtimes'][:], g['sizes']):
        st = os.stat(osp.join(run_dir, name))
        if (st.st_size != size) or (st.st_mtime != mtime):
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
        run_nr = None

    if run_nr is not None:
        fname = '%s_%s.h5' % (raw_proc, run_nr)
        prop_scratch = osp.join(
            SCRATCH_ROOT_DIR, instr, cycle, prop, 'scratch'
        )
        if osp.isdir(prop_scratch):
            paths.append(
                osp.join(prop_scratch, '.karabo_data_maps', fname)
            )
    return paths

def find_file_read(run_dir):
    for candidate in voview_paths_for_run(run_dir):
        if osp.isfile(candidate):
            return candidate

def find_file_valid(run_dir):
    for candidate in voview_paths_for_run(run_dir):
        if osp.isfile(candidate):
            file_acc = FileAccess(candidate)
            if check_sources(file_acc.file, run_dir):
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
        vofw = VirtualOverviewFileWriter(file_path, run)
        vofw.write()

if __name__ == '__main__':
    sys.exit(main())
