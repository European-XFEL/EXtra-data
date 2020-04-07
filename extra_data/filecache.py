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
import weakref
from collections import OrderedDict

class FileCache(object):
    """
    FileCache is a collection for opened h5 files. It keeps the number of opened files 
    under the given limit popping files accessed longest time ago.
    """
    def __init__(self, maxfiles=128):
        self._maxfiles = maxfiles
        self._cache = OrderedDict()
        
    def __del__(self):
        while len(self._cache):
            _, rf = self._cache.popitem(last=False)
            f = rf()
            if f:
                f.close()
        
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
            _, rf = self._cache.popitem(last=False)
            rf().close()
            n -= 1
        self._maxfiles = maxfiles

    def close_all(self):
        """
        Closes all cached files
        """
        while len(self._cache):
            _, rf = self._cache.popitem(last=False)
            rf().close()

    def open(self, filename):
        """
        Returns the opened `h5py.File` instance from the cache.
        It opens new file only if the requested file is absent.
        If new file exceeds the *maxfiles* limit it pops file accessed most far.

        For use of the file cache, FileAccess shold use `open(filename)`
        instead of direct opening file with `h5py.File`
        """
        try:
            f = self._cache[filename]()
            self._cache.move_to_end(filename)
        except KeyError:
            if len(self._cache) >= self._maxfiles:
                _, rf = self._cache.popitem(last=False)
                rf().close()
            f = h5py.File(filename, 'r')
            self._cache[filename] = weakref.ref(f, lambda o: self._cache.pop(filename, None))
        return f
    
    def touch(self, filename):
        """
        Move the touched file to the end of the `cache`
        
        For use of the file cache, FileAccess should use `touch(filename)` every time 
        it provides the underying instance of `h5py.File` for reading.
        """
        self._cache.move_to_end(filename)

    def force_close(self, filename):
        """
        Pops file from cache and closes it
        
        Useful, if it is necessary to reopen some file for writing.
        """
        rf = self._cache.pop(filename, None)
        if rf is not None:
            rf().close()

        
import resource

def set_global_filecache(fc):
    """
    Sets new FileCache instance as a global file cache
    """
    global extra_data_filecache
    extra_data_filecache = fc
    
def get_global_filecache():
    """
    Returns FileCache instance which is used as global file cache
    """
    global extra_data_filecache
    return extra_data_filecache

def init_global_filecache():
    # Raise the limit for open files (1024 -> 4096 on Maxwell)
    nofile = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (nofile[1], nofile[1]))
    maxfiles = nofile[1] // 2
    global extra_data_filecache
    extra_data_filecache = FileCache(maxfiles)

init_global_filecache()
