"""Create & check 'virtual overview' files

These use virtual datasets to present the data from a run as a single file.
"""

import os
import sys

import h5py

from .writer import VirtualFileWriter

class VirtualOverviewFileWriter(VirtualFileWriter):
    def record_source_files(self):
        grp = self.file.create_group('.source_files')
        names, mtimes, sizes = [], [], []
        for fa in self.data.files:
            st = fa.metadata_fstat or os.stat(fa.filename)
            names.append(os.path.basename(fa.filename).encode('ascii'))
            mtimes.append(st.st_mtime)
            sizes.append(st.st_size)

        grp.create_dataset(
            'names', data=names, dtype=h5py.string_dtype(encoding='ascii')
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

    files_now = {f for f in os.listdir(run_dir) if f.endswith('.h5')}
    files_stored = [p.decode('ascii') for p in g['names'][:]]
    if files_now != set(files_stored):
        return False

    for name, mtime, size in zip(files_stored, g['mtimes'][:], g['sizes']):
        st = os.stat(os.path.join(run_dir, name))
        if (st.st_size != size) or (st.st_mtime != mtime):
            return False

    return True

def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    ap.add_argument('run_dir')
    ap.add_argument('overview_file')
    args = ap.parse_args(argv)

    if args.check:
        with h5py.File(args.overview_file, 'r') as f:
            ok = check_sources(f, args.run_dir)
        if ok:
            print("Source files match, overview file can be used")
        else:
            print("Source files don't match, overview file outdated")
            return 1
    else:
        from . import RunDirectory
        run = RunDirectory(args.run_dir)
        vofw = VirtualOverviewFileWriter(args.overview_file, run)
        vofw.write()

if __name__ == '__main__':
    sys.exit(main())
