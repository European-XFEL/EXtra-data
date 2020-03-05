"""
File cache implementation

File cache provides the collection for open h5 files 
which keeps the number of opened files under the given limit
popping the files that were accessed the longest time ago

Implementation based on the OrderedDict

The module provides global instance of file cache for using
across entire process
"""
import h5py
from collections import OrderedDict

class FileRef(object):
    __slots__ = 'nref', 'fh'


class FileCache(object):
    """
    FileCache is a collection for opened h5 files. It keeps the number of opened files 
    under the given limit popping files accessed longest time ago.
    """
    def __init__(self, maxfiles=128):
        self._maxfiles = maxfiles
        self._cache = OrderedDict()
        
    def __del__(self):
        self.clean()
        
    @property
    def maxfiles(self):
        return self._maxfiles
    
    @maxfiles.setter
    def maxfiles(self, maxfiles):
        """
        Set the new file limit and pops redundant files
        """
        n = len(self._cache)
        while n > maxfiles:
            p, f = self._cache.popitem(last=False)
            f.fh.close()
            n -= 1
        self._maxfiles = maxfiles

    def clean(self):
        """
        Closes all cached files
        """
        while len(self._cache):
            p, f = self._cache.popitem(last=False)
            f.fh.close()

    def open(self, filename):
        """
        Returns the opened `h5.File` instance from the cache.
        It opens new file only if the requested file is absent.
        If new file exceeds the *maxfiles* limit it pops file accessed most far.

        For use of the file cache, FileAccess shold use `get_or_open(filename)`
        instead of direct opening file with h5py
        """
        try:
            f = self._cache[filename]
            f.nref += 1
            self._cache.move_to_end(filename)
        except KeyError:
            if len(self._cache) >= self._maxfiles:
                rp, rf = self._cache.popitem(last=False)
                rf.fh.close()
            f = FileRef()
            f.fh = h5py.File(filename, 'r')
            f.nref = 1
            self._cache[filename] = f
        return f.fh
    
    def touch(self, filename):
        """
        Move the touched file to the end of the `cache`
        
        For use of the file cache, FileAccess should use `touch(filename)` every time 
        it provides the underying instance of h5File for reading.
        """
        self._cache.move_to_end(filename)

    def close(self, filename):
        f = self._cache.get(filename, None)
        if f is not None:
            if f.nref <= 1:
                f.fh.close()
                del self._cache[filename]
            else:
                f.nref -= 1
        
    def force_close(self, filename):
        """
        Pops file from cache and closes it
        
        Useful, if it is necessary to reopen some file for writing.
        """
        f = self._cache.pop(filename, None)
        if f is not None:
            f.fh.close()

        
import resource

def set_global_filecache():
    nofile_rlimits = resource.getrlimit(resource.RLIMIT_NOFILE)
    maxfiles = nofile_rlimits[0] // 2
    global _extra_data_file_cache
    _extra_data_file_cache = FileCache(maxfiles)

def get_global_filecache():
    """
    Returns the global instance of FileCache
    """
    try:
        return _extra_data_file_cache
    except NameError:
        set_global_filecache()
        return _extra_data_file_cache

