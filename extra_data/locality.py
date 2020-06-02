"""
Tools to check a file locality at EuXFEL

May be used to avoiding hangs on reading files from dCache
if they are not available or stored only on tape
"""
import os
import sys
from collections import defaultdict
import multiprocessing as mp

UNAVAIL = 1
ONTAPE = 2
ONDISK = 4
ANY = 7

DC_LOC_RESP = {
    'UNAVAILABLE': UNAVAIL,
    'NEARLINE': ONTAPE,
    'ONLINE': ONDISK,
    'ONLINE_AND_NEARLINE': ONTAPE | ONDISK,
    'NOT_ON_DCACHE': ONDISK,
}
LOCMSG = {
    0: 'Unknown locality',
    1: 'Unavailable',
    2: 'Only on tape',
    4: 'On disk',
    6: 'On disk',
}

def get_locality(path):
    """ Returns locality of the file (path) """
    basedir, filename = os.path.split(path)
    dotcmd = os.path.join(basedir, '.(get)({})(locality)'.format(filename))
    try:
        with open(dotcmd, 'r') as f:
            return path, f.read().strip()
    except FileNotFoundError:
        return path, 'NOT_ON_DCACHE'
    
def list_locality(files):
    """ Returns locality of the list of files """
    with mp.Pool() as p:
        yield from p.imap_unordered(get_locality, files)

def print_counts(fpart):
    """ Prints the counters of different localities """
    n_ondisk = len(fpart['NOT_ON_DCACHE']) + len(fpart['ONLINE_AND_NEARLINE']) + len(fpart['ONLINE'])
    n_ontape = len(fpart['NEARLINE'])
    n_unavail = len(fpart['UNAVAILABLE'])
    print(f"{n_ondisk} on disk, {n_ontape} only on tape, {n_unavail} unavailable    ", end='\r')
    
def silent(fpart):
    """ Prints nothing """
    pass

def partition(files, cb_disp=silent):
    """ Partition files by locality """
    fpart = defaultdict(list)
    for path, loc in list_locality(files):
        fpart[loc].append(path)
        cb_disp(fpart)
    return fpart

def lc_match(files, accept=ONDISK):
    """ Returns files which has accepted locality """
    filtered = []
    for path, loc in list_locality(files):
        code = DC_LOC_RESP.get(loc, 0)
        if code & accept:
            filtered.append(path)
        else:
            print(f"Skipping file {path}", file=sys.stderr)
            print(f"  ({LOCMSG[loc]})", file=sys.stderr)
            
    return filtered
        
    
def lc_any(files):
    """ Returns all files, does nothing """
    return files

def lc_ondisk(files):
    """Returns files on disk, excluding any which would be read from tape"""
    return lc_match(files, ONDISK)

def lc_avail(files):
    """Returns files which are available on disk or tape

    Excludes files which dCache reports are unavailable.
    """
    return lc_match(files, ONTAPE | ONDISK)

def check_dir(basedir):
    """ Check basedir and prints results """
    ls = ( os.path.join(basedir, f) for f in os.listdir(basedir) )
    files = [ f for f in ls if os.path.isfile(f) ]
    
    print(f"Checking {len(files)} files in {basedir}")
    fp = partition(files, print_counts)
    print("")
    
    if fp['NEARLINE']:
        print("Only on tape:")
        for file in sorted(fp['NEARLINE']):
            print(f"  {file}")
    
    if fp['UNAVAILABLE']:
        print("Unavailable:")
        for file in sorted(fbl['UNAVAILABLE']):
            print(f"  {file}")
    
    unknown_locality = set(fp) - set(DC_LOC_RESP)
    if unknown_locality:
        print("Unknown locality:", unknown_keys)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f'Usage: {sys.argv[0]} <directory>')
        exit(1)
        
    check_dir(sys.argv[1])
