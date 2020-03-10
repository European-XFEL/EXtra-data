"""                                                                                                                             Tool to check a file locality at EuXFEL

Maybe used to avoid hangs on reading files from dCache
if they are not available or stored on tape
"""
import sys
import os
import errno
import fnmatch
import multiprocessing as mp
import psutil


MW_DCACHE_MOUNT='/pnfs/xfel.eu/exfel'

NONE = 0x1
GPFS = 0x2
DC_ONLINE = 0x4
DC_NEARLINE = 0x8
DC_ANY = 0xC
UNKNOWN = 0

FAST = 0x6
ANY = 0xF

DC_LOC_RESP = {
    'UNAVAILABLE': NONE,
    'NEARLINE': DC_NEARLINE,
    'ONLINE': DC_ONLINE,
    'ONLINE_AND_NEARLINE': DC_ANY,
}

LAB = {
    1: 'unavailable',
    2: 'fast: GPFS',
    4: 'slow: PNFS, on tape',
    8: 'fast: PNFS, online',
    12: 'fast: PNFS, online',
}

class FileTemporarilyUnavailable(OSError):
    pass

def isondcache(path, dcache_mount=MW_DCACHE_MOUNT):
    """ Return flag that file on dCache """
    return os.path.realpath(os.path.abspath(path)).startswith(dcache_mount)

def get_locality(path):
    """ Returns locality of the file (path) """
    
    if not isondcache(path):
        return GPFS

    bdir, fn = os.path.split(path)
    cmd = os.path.join(bdir, f".(get)({fn})(locality)")
    with open(cmd, 'rt') as f:
        loc = DC_LOC_RESP.get(f.read().strip(), UNKNOWN)
    return loc

def check(path, accpt_loc):
    """ Raises exception unless file (path) has accepted locality (accpt_loc) """
    loc = get_locality(path)
    if not loc & ANY:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), path)
    if not loc & accpt_loc:
        raise FileTemporarilyUnavailable(errno.EAGAIN, os.strerror(errno.EAGAIN), path)


def get_locality_worker(path):
    return path, get_locality(path)
def init_loclaity_worker():
    pass

def lsdir(path, pattern='*', accpt_loc=ANY, nproc=None):
    """
    Returns the list of files in the directory which are has accepted localilty (accpt_loc)
    
    Runs in parallel unless nproc == 1
    """
    
    if nproc is None:
        files = [ os.path.join(path, fn) for fn in os.listdir(path) if fnmatch.fnmatch(fn, pattern) ]
        nproc = min(len(psutil.Process().cpu_affinity()), len(files), 10)
    else:
        files = ( os.path.join(path, fn) for fn in os.listdir(path) if fnmatch.fnmatch(fn, pattern) )
    if nproc > 1:
        with mp.Pool(nproc, initializer=init_loclaity_worker) as pool:
            r = pool.imap_unordered(get_locality_worker, files)
            yield from ( (ffn, loc) for ffn, loc in r if loc & accpt_loc )
    else:
        r = ( (fn, get(fn)) for fn in files )
        yield from ( (ffn, loc) for ffn, loc in r if loc & accpt_loc )
        
        
def check_directory(path, display_except = ANY, nproc=None):
    """ Check the directory and print files which do not have accepted locality """
    lenp = len(path)
    nflt, nacc = 0, 0
    for f, l in lsdir(path, "*.h5", nproc=nproc):
        if not (l & display_except):
            nflt += 1
            print("{:20} {}".format(LAB[l], f[lenp:]))
        else:
            nacc += 1
            
    print("accepted: {}, filtered: {}".format(nacc, nflt))
        
        
if __name__ == '__main__':
    
    if len (sys.argv) < 2:
        print('Usage: {} dir1 dir2 ... dirN'.format(sys.argv[0]))
    for argv in sys.argv[1:]:
        print('* ', argv)
        try:
            check_directory(argv, FAST)
        except Exception as e:
            print(str(e))
            