Reading data files
==================

.. _opening-files:

Opening files
-------------

You will normally access data from a run, which is stored as a directory
containing HDF5 files. You can open a run using :func:`RunDirectory` with the
path of the directory, or using :func:`open_run` with the proposal number and
run number to look up the standard data paths on the Maxwell cluster.

.. module:: extra_data

.. autofunction:: RunDirectory

.. autofunction:: open_run

You can also open a single file. The methods described below all work for either
a run or a single file.

.. autofunction:: H5File

See :ref:`suspect-trains` for more details about the ``inc_suspect_trains``
parameter.

Data structure
--------------

A run (or file) contains data from various *sources*, each of which has *keys*.
For instance, ``SA1_XTD2_XGM/XGM/DOOCS`` is one source, for an 'XGM' device
which monitors the beam, and its keys include ``beamPosition.ixPos`` and
``beamPosition.iyPos``.

European XFEL produces ten *pulse trains* per second, each of which can contain
up to 2700 X-ray pulses. Each pulse train has a unique train ID, which is used
to refer to all data associated with that 0.1 second window.

.. class:: DataCollection

   .. attribute:: train_ids

      A list of the train IDs included in this data.
      The data recorded may not be the same for each train.

   .. attribute:: control_sources

      A set of the control source names in this data, in the format
      ``"SA3_XTD10_VAC/TSENS/S30100K"``. Control data is always recorded
      exactly once per train.

   .. attribute:: instrument_sources

      A set of the instrument source names in this data,
      in the format ``"FXE_DET_LPD1M-1/DET/15CH0:xtdf"``.
      Instrument data may be recorded zero to many times per train.

   .. attribute:: all_sources

      A set of names for both instrument and control sources.
      This is the union of the two sets above.

   .. automethod:: keys_for_source

   .. automethod:: get_data_counts

   .. automethod:: train_timestamps

   .. automethod:: run_metadata

      .. versionadded:: 1.6

   .. automethod:: info

.. _data-by-source-and-key:

Getting data by source & key
----------------------------

Selecting a source in a run gives a :class:`.SourceData` object.
You can use this to find keys belonging to that source::

    xgm = run['SPB_XTD9_XGM/DOOCS/MAIN']
    xgm.keys()  # List the available keys
    beam_x = xgm['beamPosition.ixPos']  # Get a KeyData object

Selecting a single source & key in a run gives a :class:`.KeyData` object.
You can get the data from this in various forms with the methods described
below, e.g.::

    xgm_intensity = run['SA1_XTD2_XGM/XGM/DOOCS:output', 'data.intensityTD'].xarray()

.. class:: KeyData

   .. attribute:: dtype

      The NumPy dtype for this data. This indicates whether it contains
      integers or floating point numbers, and how many bytes of memory each
      number needs.

   .. attribute:: ndim

      The number of dimensions the data has. All data has at least 1 dimension
      (time). A sequence of 2D images would have 3 dimensions.

   .. autoattribute:: shape

   .. attribute:: entry_shape

      The shape of a single entry in the data, e.g. a single frame from a
      camera. This is equivalent to ``key.shape[1:]``, but may be quicker than
      that.

   .. automethod:: data_counts

   .. automethod:: ndarray

   .. automethod:: series

   .. automethod:: xarray

     .. seealso::
       `xarray documentation <https://xarray.pydata.org/en/stable/indexing.html>`__
         How to use the arrays returned by :meth:`DataCollection.get_array`

       :doc:`xpd_examples`
         Examples using xarray & pandas with EuXFEL data

   .. automethod:: dask_array

    .. seealso::
       `Dask Array documentation <https://docs.dask.org/en/latest/array.html>`__
         How to use the objects returned by :meth:`DataCollection.get_dask_array`

       :doc:`dask_averaging`
         An example using Dask with EuXFEL data

   .. automethod:: train_id_coordinates

   .. automethod:: select_trains
   
   .. automethod:: split_trains

      .. versionadded:: 1.7
      
   .. automethod:: drop_empty_trains
   
      .. versionadded:: 1.7

   .. automethod:: as_single_value
   
      .. versionadded:: 1.9


The run or file object (a :class:`DataCollection`) also has methods to load
data by sources and keys. :meth:`get_array`, :meth:`get_dask_array` and
:meth:`get_series` are directly equivalent to the options above, but other
methods offer extra capabilities.

.. class:: DataCollection
   :noindex:

   .. automethod:: get_array

   .. automethod:: get_dask_array

   .. automethod:: get_series

   .. automethod:: get_dataframe

      .. seealso::
        `pandas documentation <https://pandas.pydata.org/pandas-docs/stable/>`__
          How to use the objects returned by :meth:`~.get_series` and
          :meth:`~.get_dataframe`

        :doc:`xpd_examples`
          Examples using xarray & pandas with EuXFEL data

   .. automethod:: get_virtual_dataset

      .. seealso::
        :doc:`parallel_example`

   .. automethod:: get_run_value

      .. versionadded:: 1.6

   .. automethod:: get_run_values

      .. versionadded:: 1.6

.. _data-by-train:

Getting data by train
---------------------

Some kinds of data, e.g. from AGIPD, are too big to load a whole run into
memory at once. In these cases, it's convenient to load one train at a time.

If you want to do this for just one source & key with :class:`KeyData` methods,
like this::

    for tid, arr in run['SA1_XTD2_XGM/XGM/DOOCS:output', 'data.intensityTD'].trains():
        ...

.. class:: KeyData
   :noindex:

   .. automethod:: trains

   .. automethod:: train_from_id

   .. automethod:: train_from_index

To work with multiple modules of the same detector, see :doc:`agipd_lpd_data`.

You can also get data by train for multiple sources and keys together from a run
or file object.
It's always a good idea to select the data you're interested in, either using
:meth:`~.DataCollection.select`, or the ``devices=`` parameter. If you don't,
they will read data for all sources in the run, which may be very slow.

.. class:: DataCollection
   :noindex:

   .. automethod:: trains

   .. automethod:: train_from_id

   .. automethod:: train_from_index

Selecting & combining data
--------------------------

These methods all return a new :class:`DataCollection` object with the selected
data, so you use them like this::

    sel = run.select("*/XGM/*")
    # sel includes only XGM sources
    # run still includes all the data

.. class:: DataCollection
   :noindex:

   .. automethod:: select

   .. automethod:: deselect

   .. automethod:: select_trains

   .. automethod:: split_trains

      .. versionadded:: 1.7

   .. automethod:: union

Writing selected data
---------------------

.. class:: DataCollection
   :noindex:

   .. automethod:: write

   .. automethod:: write_virtual

Missing data
------------

What happens if some data was not recorded for a given train?

Control data is duplicated for each train until it changes.
If the device cannot send changes, the last values will be recorded for each
subsequent train until it sends changes again.
There is no general way to distinguish this scenario from values which
genuinely aren't changing.

Parts of instrument data may be missing from the file. These will also be
missing from the data returned by ``extra_data``:

- The train-oriented methods :meth:`~.DataCollection.trains`,
  :meth:`~.DataCollection.train_from_id`, and
  :meth:`~.DataCollection.train_from_index` give you dictionaries keyed by
  source and key name. Sources and keys are only included if they have
  data for that train.
- :meth:`~.DataCollection.get_array`, and
  :meth:`~.DataCollection.get_series` skip over trains which are missing data.
  The indexes on the returned DataArray or Series objects link the returned
  data to train IDs. Further operations with xarray or pandas may drop
  misaligned data or introduce fill values.
- :meth:`~.DataCollection.get_dataframe` includes rows for which any column has
  data. Where some but not all columns have data, the missing values are filled
  with ``NaN`` by pandas' `missing data handling
  <https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html>`__.

Missing data does not necessarily mean that something has gone wrong:
some devices send data at less than 10 Hz (the train rate), so they always
have gaps between updates.

Data problems
-------------

If you encounter problems accessing data with ``extra_data``, there may be
problems with the data files themselves. Use the ``extra-data-validate``
command to check for this (see :doc:`validation`).

Here are some problems we've seen, and possible solutions or workarounds:

- Indexes point to data beyond the end of datasets:
  this has previously been caused by bugs in the detector calibration pipeline.
  If you see this in calibrated data (in the ``proc/`` folder),
  ask for the relevant runs to be re-calibrated.
- Train IDs are not strictly increasing:
  issues with the timing system when the data is recorded can create an
  occasional train ID which is completely out of sequence.
  Usually it seems to be possible to ignore this and use the remaining data,
  but if you have any issues, please let us know.

  - In one case, a train ID had the maximum possible value (2\ :sup:`64` - 1),
    causing :meth:`~.info` to fail. You can select everything except this train
    using :meth:`~.DataCollection.select_trains`::

        from extra_data import by_id
        sel = run.select_trains(by_id[:2**64-1])

If you're having problems with extra_data, you can also try searching
`previously reported issues <https://github.com/European-XFEL/EXtra-data/issues?q=is%3Aissue>`_
to see if anyone has encountered similar symptoms.

.. _run-map-caching:

Cached run data maps
--------------------

When you open a run in extra_data, it needs to know what data is in each file.
Each file has metadata describing its contents, but reading this from every file
is slow, especially on GPFS. extra_data therefore tries to cache this
information the first time a run is opened, and reuse it when opening that run
again.

This should happen automatically, without the user needing to know about it.
You only need these details if you think caching may be causing problems.

- Caching is triggered when you use :func:`RunDirectory` or :func:`open_run`.
- There are two possible locations for the cached data map:

  - In the run directory: ``(run dir)/karabo_data_map.json``.
  - In the proposal scratch directory:
    ``(proposal dir)/scratch/.karabo_data_maps/raw_r0032.json``.
    This will normally be the one used on Maxwell, as users can't write to the
    run directory.

- The format is a JSON array, with an object for each file in the run.

  - This holds the list of train IDs in the file, and the lists of control and
    instrument sources.
  - It also stores the file size and last modified time of each data file, to
    check if the file has changed since the cache was created. If either of
    these attributes doesn't match, extra_data ignores the cached information
    and reads the metadata from the HDF5 file.

- If any file in the run wasn't listed in the data map, or its entry was
  outdated, a new data map is written automatically. It tries the same two
  locations described above, but it will continue without error if it can't
  write to either.

JSON was chosen as it can be easily inspected manually, and it's reasonably
efficient to load the entire file.

Issues reading archived data
----------------------------

Files at European XFEL storage migrate over time from GPFS (designed for fast access) to PNFS (designed for archiving). The data 
on PNFS is usually always available for reading. But sometimes, this may require
staging from the tape to disk. If there is a staging queue, the operation can take
an indefinitely long time (days or even weeks) and any IO operations will be 
blocked for this time.

To determine the files which require staging or are lost, use the script::

    extra-data-locality <run directory>

It returns a list of files which are currently located only on slow media for some
reasons and, separately, any which have been lost.

If the files are not essential for analysis, then they can be filtered out using 
filter :func:`lc_ondisk` from :mod:`extra_data.locality`::

    from extra_data.locality import lc_ondisk
    run = open_run(proposal=700000, run=1, file_filter=lc_ondisk)

``file_filter`` must be a callable which takes a list as a single argument and
returns filtered list.

**Note: Reading the file locality on PNFS is an expensive operation.
Use it only as a last resort.**

If you find any files which are located only on tape or unavailable, please let know to
`ITDM <mailto:it-support@xfel.eu>`_. If you need these files for analysis mentioned
that explicitly.

.. _suspect-trains:

'Suspect' train IDs
-------------------

In some cases (especially with AGIPD data), some train IDs appear to be recorded
incorrectly, breaking the normal assumption that train IDs are in increasing
order. EXtra-data will exclude these trains by default, but you can try to
access them by passing ``inc_suspect_trains=True`` when :ref:`opening a file
or run <opening-files>`. Some features may not work correctly if you do this.

In newer files (format version 1.0 or above), trains are considered suspect
where their ``INDEX/flag`` entry is 0. This indicates that the DAQ received the train ID
from a device before it received it from a time server. This appears to be a reliable
indicator of erroneous train IDs.

In older files without ``INDEX/flag``, EXtra-data tries to guess which trains
are suspect. The good trains should make an increasing sequence, and it tries to
exclude as few trains as possible to achieve this. If something goes wrong with
this guessing, try using ``inc_suspect_trains=True`` to avoid it.
Please let us know (da-support@xfel.eu) if you need to do this.
