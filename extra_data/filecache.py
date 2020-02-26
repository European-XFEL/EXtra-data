import h5py

class CachedFile(object):
    def __init__(self, filename, prio, cache_instance=None):
        self._filename = filename
        self._fc_prio = prio
        self._fc_pos = -1
        self._file = h5py.File(filename, 'r')
        self._cache = cache_instance if cache_instance is not None else get_global_filecache()
        

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None
            self._fc_pos = -1
        
    def __lt__(self, other):
        return self._fc_prio < other._fc_prio

    @property
    def file(self):
        if self._file is not None:
            self._cache._touch(self._fc_pos)
        return self._file
    
class DummyCachedFile(object):
    file = None
        
        
class FileCache(object):
    dummy = DummyCachedFile
    
    def __init__(self, maxfiles=128):
        self._maxfiles = maxfiles
        self.pq = []
        self.cat = {}
        self.counter = 0
        
    @property
    def maxfiles(self):
        return self._maxfiles
    
    @maxfiles.setter
    def maxfiles(self, maxfiles):
        n = len(self.pq)
        while n > maxfiles:
            f = self._pop()
            del self.cat[f._filename]
            f.close()
            n -= 1
        self._maxfiles = maxfiles
        
    def clean(self):
        self.cat = {}
        self.counter = 0
        while self.pq:
            f = self._pop()
            f.close()

    def get_or_open(self, filename):
        try:
            f = self.cat[filename]
            pos = f._fc_pos
            f._fc_prio = self.counter
            self._siftup(pos)
        except:
            f = CachedFile(filename, self.counter)
            n = len(self.pq)
            if n < self._maxfiles:
                f._fc_pos = n
                self.pq.append(f)
                self._siftdown(0, n)
            else:
                redundant = self.pq[0]
                del self.cat[redundant._filename]
                redundant.close()

                f._fc_pos = 0
                self.pq[0] = f
                self._siftup(0)

            self.cat[filename] = f

        self.counter += 1
        
        return f
    
    def _touch(self, pos):
        self.pq[pos]._fc_prio = self.counter
        self._siftup(pos)
        self.counter += 1
    
    def _pop(self):
        last = self.pq.pop()    # raises appropriate IndexError if heap is empty
        if self.pq:
            ret = self.pq[0]
            last._fc_pos = 0
            self.pq[0] = last
            self._siftup(0)
            return ret
        
        return last
        

    def _siftdown(self, startpos, pos):
        newitem = self.pq[pos]
        # Follow the path to the root, moving parents down until finding a place
        # newitem fits.
        while pos > startpos:
            parentpos = (pos - 1) >> 1
            parent = self.pq[parentpos]
            if newitem < parent:
                parent._fc_pos = pos
                self.pq[pos] = parent
                pos = parentpos
                continue
            break
        newitem._fc_pos = pos
        self.pq[pos] = newitem

    def _siftup(self, pos):
        endpos = len(self.pq)
        startpos = pos
        newitem = self.pq[pos]
        # Bubble up the smaller child until hitting a leaf.
        childpos = 2*pos + 1    # leftmost child position
        while childpos < endpos:
            # Set childpos to index of smaller child.
            rightpos = childpos + 1
            if rightpos < endpos and not self.pq[childpos] < self.pq[rightpos]:
                childpos = rightpos
            # Move the smaller child up.
            item = self.pq[childpos]
            item._fc_pos = pos
            self.pq[pos] = item
            pos = childpos
            childpos = 2*pos + 1
        # The leaf at pos is empty now.  Put newitem there, and bubble it up
        # to its final resting place (by sifting its parents down).
        newitem._fc_pos = pos
        self.pq[pos] = newitem
        self._siftdown(startpos, pos)
        
import resource
import atexit

def set_global_filecache():
    nofile_rlimits = resource.getrlimit(resource.RLIMIT_NOFILE)
    maxfiles = nofile_rlimits[0] >> 1
    global _extra_data_file_cache
    _extra_data_file_cache = FileCache(maxfiles)

def get_global_filecache():
    return _extra_data_file_cache

set_global_filecache()

@atexit.register
def clean_global_filecache():
    _extra_data_file_cache.clean()
