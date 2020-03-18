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

    def __init__(self, fh):
        self.fh = fh
        self.nref = 1

class FileCache(object):
    """
    FileCache is a collection for opened h5 files. It keeps the number of opened files 
    under the given limit popping files accessed longest time ago.
    """
    def __init__(self, maxfiles=128):
        self._maxfiles = maxfiles
        self._cache = OrderedDict()
        
    def __del__(self):
        self.close_all()
        
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

    def close_all(self):
        """
        Closes all cached files
        """
        while len(self._cache):
            p, f = self._cache.popitem(last=False)
            f.fh.close()

    def open(self, filename):
        """
        Returns the opened `h5py.File` instance from the cache.
        It opens new file only if the requested file is absent.
        If new file exceeds the *maxfiles* limit it pops file accessed most far.

        For use of the file cache, FileAccess shold use `open(filename)`
        instead of direct opening file with `h5py.File`
        """
        try:
            f = self._cache[filename]
            f.nref += 1
            self._cache.move_to_end(filename)
        except KeyError:
            if len(self._cache) >= self._maxfiles:
                rp, rf = self._cache.popitem(last=False)
                rf.fh.close()
            fh = h5py.File(filename, 'r')
            f = FileRef(fh)
            self._cache[filename] = f
        return f.fh
    
    def touch(self, filename):
        """
        Move the touched file to the end of the `cache`
        
        For use of the file cache, FileAccess should use `touch(filename)` every time 
        it provides the underying instance of `h5py.File` for reading.
        """
        self._cache.move_to_end(filename)

    def close(self, filename):
        """
        Closes the underlying file if called the same times as `open(filename)`
        Otherwise, just decreases the counter.
        """
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

def init_extra_data_filecache():
    nofile = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (nofile[1], nofile[1]))
    maxfiles = nofile[1] // 2
    global extra_data_filecache
    extra_data_filecache = FileCache(maxfiles)

init_extra_data_filecache()
