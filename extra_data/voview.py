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
            names.append(fa.filename.encode('utf-8'))
            mtimes.append(st.st_mtime)
            sizes.append(st.st_size)

        grp.create_dataset('names', data=names, dtype=h5py.string_dtype())
        grp.create_dataset('mtimes', data=mtimes, dtype='f4')
        grp.create_dataset('sizes', data=sizes, dtype='u8')

    def write(self):
        self.record_source_files()
        super().write()


if __name__ == '__main__':
    if len(sys.argv) < 3:
        sys.exit("Usage: python -m extra_data.voverview run_dir filename")

    from . import RunDirectory
    run = RunDirectory(sys.argv[1])
    vofw = VirtualOverviewFileWriter(sys.argv[2], run)
    vofw.write()

