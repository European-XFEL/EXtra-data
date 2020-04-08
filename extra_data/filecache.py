"""
File cache implementation

File cache provides the collection for open h5 files 
which keeps the number of opened files under the given limit
popping the files that were accessed the longest time ago

Implementation based on the OrderedDict

The module provides global instance of file cache for using
across entire process
"""
from collections import OrderedDict
from weakref import WeakValueDictionary


# Track all FileAccess objects - {path: FileAccess}
file_access_registry = WeakValueDictionary()


class FileCache(object):
    """
    FileCache is a collection for opened h5 files. It keeps the number of opened files 
    under the given limit popping files accessed longest time ago.
    """
    def __init__(self, maxfiles=128):
        self._maxfiles = maxfiles
        # We don't use the values, but OrderedDict is a handy as a queue
        # with efficient removal of entries by key.
        self._cache = OrderedDict()
        
    @property
    def maxfiles(self):
        return self._maxfiles
    
    @maxfiles.setter
    def maxfiles(self, maxfiles):
        """Set the new file limit and closes files over the limit"""
        self._maxfiles = maxfiles
        self.close_old_files()

    def _check_files(self):
        # Discard entries from self._cache if their FileAccess no longer exists
        self._cache = OrderedDict.fromkeys(
            path for path in self._cache if path in file_access_registry
        )

    def n_open_files(self):
        self._check_files()
        return len(self._cache)

    def close_old_files(self):
        if len(self._cache) <= self.maxfiles:
            return

        # Now check how many paths still have an existing FileAccess object
        n = self.n_open_files()
        while n > self.maxfiles:
            path, _ = self._cache.popitem(last=False)
            file_access = file_access_registry.get(path, None)
            if file_access is not None:
                file_access.close()
            n -= 1
    
    def touch(self, filename):
        """
        Add/move the touched file to the end of the `cache`.

        If adding a new file takes it over the limit of open files, another file
        will be closed.
        
        For use of the file cache, FileAccess should use `touch(filename)` every time 
        it provides the underying instance of `h5py.File` for reading.
        """
        if filename in self._cache:
            self._cache.move_to_end(filename)
        else:
            self._cache[filename] = None
            self.close_old_files()

    def closed(self, filename):
        """Discard a closed file from the cache
        """
        self._cache.pop(filename, None)

        
import resource

def init_extra_data_filecache():
    # Raise the limit for open files (1024 -> 4096 on Maxwell)
    nofile = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (nofile[1], nofile[1]))
    maxfiles = nofile[1] // 2
    global extra_data_filecache
    extra_data_filecache = FileCache(maxfiles)

init_extra_data_filecache()
