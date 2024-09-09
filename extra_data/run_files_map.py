import json
import logging
import os
import os.path as osp
import numpy as np
import re
from tempfile import mkstemp
import time

SCRATCH_ROOT_DIR = "/gpfs/exfel/exp/"

log = logging.getLogger(__name__)

def atomic_dump(obj, path, **kwargs):
    """Write JSON to a file atomically

    This aims to avoid garbled files from multiple processes writing the same
    cache. It doesn't try to protect against e.g. sudden power failures, as
    forcing the OS to flush changes to disk may hurt performance.
    """
    dirname, basename = osp.split(path)
    fd, tmp_filename = mkstemp(dir=dirname, prefix=basename)
    try:
        with open(fd, 'w') as f:
            json.dump(obj, f, **kwargs)
    except:
        os.unlink(tmp_filename)
        raise

    os.replace(tmp_filename, path)


class RunFilesMap:
    """Cached data about HDF5 files in a run directory

    Stores the train IDs and source names in each file, along with some
    metadata to check that the cache is still valid. The cached information
    can be stored in:

    - (run dir)/karabo_data_map.json
    - (proposal dir)/scratch/.karabo_data_maps/raw_r0032.json
    """
    cache_file = None

    expected_cache_keys = frozenset({
        'train_ids',
        'control_sources',
        'instrument_sources',
        'suspect_train_indices',
        'legacy_sources',
    })

    def __init__(self, directory):
        self.directory = osp.abspath(directory)
        self.dir_stat = os.stat(self.directory)
        self.files_data = {}

        self.candidate_paths = self.map_paths_for_run(directory)

        self.load()

    def map_paths_for_run(self, directory):
        paths = [osp.join(directory, 'karabo_data_map.json')]
        # After resolving symlinks, data on Maxwell is stored in either
        # GPFS, e.g. /gpfs/exfel/d/proc/SCS/201901/p002212  or
        # dCache, e.g. /pnfs/xfel.eu/exfel/archive/XFEL/raw/SCS/201901/p002212
        # On the online cluster the resolved path stay:
        #   /gpfs/exfel/exp/inst/cycle/prop/(raw|proc)/run
        maxwell_match = re.match(
            #     raw/proc  instr  cycle prop   run
            r'.+/(raw|proc|red|open)/(\w+)/(\w+)/(p\d+)/(r\d+)/?$',
            os.path.realpath(directory)
        )
        online_match = re.match(
            #     instr cycle prop   raw/proc   run
            r'^.+/(\w+)/(\w+)/(p\d+)/(raw|proc)/(r\d+)/?$',
            os.path.realpath(directory)
        )

        if maxwell_match or online_match:
            if maxwell_match:
                raw_proc, instr, cycle, prop, run_nr = maxwell_match.groups()
            else:
                instr, cycle, prop, raw_proc, run_nr = online_match.groups()

            fname = '%s_%s.json' % (raw_proc, run_nr)
            prop_scratch = osp.join(
                SCRATCH_ROOT_DIR, instr, cycle, prop, 'scratch'
            )
            if osp.isdir(prop_scratch):
                paths.append(
                    osp.join(prop_scratch, '.karabo_data_maps', fname)
                )

        return paths

    def load(self):
        """Load the cached data

        This drops invalid or incomplete cache entries.
        """
        loaded_data = []
        t0 = time.monotonic()

        paths_mtimes = []

        for path in self.candidate_paths:
            try:
                st = os.stat(path)
                paths_mtimes.append((path, st.st_mtime))
            except (FileNotFoundError, PermissionError):
                pass

        # Try the newest found file (greatest mtime) first
        for path, _ in sorted(paths_mtimes, key=lambda x: x[1], reverse=True):
            try:
                with open(path) as f:
                    loaded_data = json.load(f)

                self.cache_file = path
                log.debug("Loaded cached files map from %s", path)
                break
            except (FileNotFoundError, PermissionError, json.JSONDecodeError,):
                pass

        for info in loaded_data:
            filename = info['filename']
            try:
                st = os.stat(osp.join(self.directory, filename))
            except OSError:
                continue
            if self._cache_info_valid(info, st):
                self.files_data[filename] = info

        if loaded_data:
            dt = time.monotonic() - t0
            log.debug("Loaded cached files map in %.2g s", dt)

    @classmethod
    def _cache_info_valid(cls, info, file_stat: os.stat_result):
        # Ignore the cached info if the file size or mtime have changed, or
        # if it is missing expected keys (likely keys added more recently).
        return ((file_stat.st_mtime == info['mtime'])
                and (file_stat.st_size == info['size'])
                and cls.expected_cache_keys.issubset(info.keys()))

    def is_my_directory(self, dir_path):
        return osp.samestat(os.stat(dir_path), self.dir_stat)

    def get(self, path):
        """Get cache entry for a file path

        Returns a dict or None
        """
        dirname, fname = osp.split(osp.abspath(path))
        if self.is_my_directory(dirname) and (fname in self.files_data):
            d = self.files_data[fname]
            res = {
                'train_ids': np.array(d['train_ids'], dtype=np.uint64),
                'control_sources': frozenset(d['control_sources']),
                'instrument_sources': frozenset(d['instrument_sources']),
                'legacy_sources': dict(d['legacy_sources']),
            }
            res['flag'] = flag = np.ones_like(d['train_ids'], dtype=np.bool_)
            flag[d['suspect_train_indices']] = 0
            return res

        return None

    def save(self, files):
        """Save the cache if needed

        This skips writing the cache out if all the data files already have
        valid cache entries. It also silences permission errors from writing
        the cache file.
        """
        need_save = False

        for file_access in files:
            dirname, fname = osp.split(osp.abspath(file_access.filename))
            if self.is_my_directory(dirname) and fname not in self.files_data:
                log.debug("Will save cached data for %s", fname)
                need_save = True

                # It's possible that the file we opened has been replaced by a
                # new one before this runs. If possible, use the stat FileAccess got
                # from the file descriptor, which will always be accurate.
                # Stat-ing the filename will almost always work as a fallback.
                if isinstance(file_access.metadata_fstat, os.stat_result):
                    st = file_access.metadata_fstat
                else:
                    log.warning("No fstat for %r, will stat name instead",
                                fname)
                    st = os.stat(file_access.filename)

                self.files_data[fname] = {
                    'filename': fname,
                    'mtime': st.st_mtime,
                    'size': st.st_size,
                    'train_ids': [int(t) for t in file_access.train_ids],
                    'control_sources': sorted(file_access.control_sources),
                    'instrument_sources': sorted(file_access.instrument_sources),
                    'legacy_sources': {k: file_access.legacy_sources[k]
                                       for k in sorted(file_access.legacy_sources)},
                    'suspect_train_indices': [
                        int(i) for i in (~file_access.validity_flag).nonzero()[0]
                    ],
                }

        if need_save:
            t0 = time.monotonic()
            save_data = [info for (_, info) in sorted(self.files_data.items())]
            for path in self.candidate_paths:
                try:
                    os.makedirs(osp.dirname(path), exist_ok=True)
                    atomic_dump(save_data, path, indent=2)
                except PermissionError:
                    continue
                else:
                    dt = time.monotonic() - t0
                    log.debug("Saved run files map to %s in %.2g s", path, dt)
                    return

            log.debug("Unable to save run files map")
