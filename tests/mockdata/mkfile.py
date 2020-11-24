import h5py
from .base import write_base_index, write_metadata

def write_file(filename, devices, ntrains, firsttrain=10000, chunksize=200,
               format_version='0.5'):
    f = h5py.File(filename, 'w')
    f.create_group('RUN')  # Add this, even if it's left empty

    write_base_index(f, ntrains, first=firsttrain, chunksize=chunksize,
                     format_version=format_version)

    data_sources = []
    for dev in devices:
        dev.ntrains = ntrains
        dev.firsttrain = firsttrain
        dev.chunksize = chunksize
        dev.write_control(f)
        dev.write_instrument(f)
        data_sources.extend(dev.datasource_ids())
    write_metadata(f, data_sources, chunksize=chunksize,
                   format_version=format_version)
    f.close()
