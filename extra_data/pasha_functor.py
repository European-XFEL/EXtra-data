
from os import getpid

import numpy as np
from pasha.functor import gen_split_slices

from . import DataCollection, SourceData, KeyData


class ExtraDataFunctor:
    """Pasha functor for EXtra-data objects.

    This functor wraps an EXtra-data DataCollection, SourceData or
    KeyData and performs the map operation over its trains. The kernel
    is passed the current train's index in the collection, the train ID
    and the data mapping (for DataCollection and SourceData) or data
    entry (for KeyData).
    """

    def __init__(self, obj):
        self.obj = obj
        self.n_trains = len(self.obj.train_ids)

        # Save PID of parent process where the functor is created to
        # close files as appropriately later on, see comment below.
        self._parent_pid = getpid()

    @classmethod
    def wrap(cls, value):
        if isinstance(value, (DataCollection, SourceData, KeyData)):
            return cls(value)

    def split(self, num_workers):
        return gen_split_slices(self.n_trains, n_parts=num_workers)

    def iterate(self, share):
        subobj = self.obj.select_trains(np.s_[share])

        # Older versions of HDF < 1.10.5 are not robust against sharing
        # a file descriptor across threads or processes. If running in a
        # different process than the functor was initially created in,
        # close all file handles inherited from the parent collection to
        # force re-opening them again in each child process.
        if getpid() != self._parent_pid:
            for f in subobj.files:
                f.close()

        index_it = range(*share.indices(self.n_trains))

        if isinstance(subobj, SourceData):
            # SourceData has no trains() iterator yet, so simulate it
            # ourselves by reconstructing a DataCollection object and
            # use its trains() iterator.
            dc = DataCollection(
                subobj.files, {subobj.source: subobj}, subobj.train_ids,
                inc_suspect_trains=subobj.inc_suspect_trains,
                is_single_run=True)
            data_it = ((train_id, data[subobj.source])
                       for train_id, data in dc.trains())
        else:
            # Use the regular trains() iterator for DataCollection and
            # KeyData
            data_it = subobj.trains()

        for index, (train_id, data) in zip(index_it, data_it):
            yield index, train_id, data
