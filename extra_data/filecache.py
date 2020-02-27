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

class H5FileDummy(object):
    """
    The fake `h5py.File` allows to unify interface for FileAccess object
    """
    class id:
        valid = 0
        
        
class FileCache(object):
    """
    FileCache is a collection for opened h5 files. It keeps the number of opened files 
    under the given limit popping files accessed longest time ago.
    """
    dummy = H5FileDummy
    
    def __init__(self, maxfiles=128):
        self._maxfiles = maxfiles
        self._cache = OrderedDict()
        
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
            rp, rf = self._cache.popitem(last=False)
            rf.close()
            n -= 1
        self._maxfiles = maxfiles
        
    def clean(self):
        """
        Closes all cached files
        """
        self.cat = {}
        self.counter = 0
        while len(self._cache):
            rp, rf = self._cache.popitem(last=False)
            rf.close()

    def get_or_open(self, filename):
        """
        Returns the opened `h5.File` instance from the cache.
        It opens new file only if the requested file is absent.
        If new file exceeds the *maxfiles* limit it pops file accessed most far.

        For use of the file cache, FileAccess shold use `get_or_open(filename)`
        instead of direct opening file with h5py
        """
        try:
            f = self._cache[filename]
            self._cache.move_to_end(filename)
        except:
            if len(self._cache) >= self._maxfiles:
                rp, rf = self._cache.popitem(last=False)
                rf.close()
            f = h5py.File(filename, 'r')
            self._cache[filename] = f
        return f
    
    def touch(self, filename):
        """
        Move the touched file to the end of the `cache`
        
        For use of the file cache, FileAccess should use `touch(filename)` every time 
        it provides the underying instance of h5File for reading.
        """
        self._cache.move_to_end(filename)
        
    def close(self, filename):
        """
        Pops file from cache and closes it
        
        Useful, if it is necessary to reopen some file for writing.
        """
        try:
            f = self._cache[filename]
            f.close()
            del self._cache[filename]
        except:
            pass
    
        
import resource
import atexit

def set_global_filecache():
    nofile_rlimits = resource.getrlimit(resource.RLIMIT_NOFILE)
    maxfiles = nofile_rlimits[0] >> 1
    global _extra_data_file_cache
    _extra_data_file_cache = FileCache(maxfiles)

def get_global_filecache():
    """
    Returns the global instance of FileCache
    """
    return _extra_data_file_cache

set_global_filecache()

@atexit.register
def clean_global_filecache():
    _extra_data_file_cache.clean()
