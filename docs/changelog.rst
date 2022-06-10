Release Notes
=============

1.12
----

2022-06-10

- :class:`.SourceData` objects now expose RUN information for control sources
  via new ``.run_value()`` and ``.run_values()`` methods, and metadata about the
  run from a new ``.run_metadata()`` method (:ghpull:`293`).
- :meth:`.KeyData.ndarray` can now read into a pre-allocated array passed
  as the ``out`` parameter (:ghpull:`307`)
- :meth:`.KeyData.xarray` can return an xarray Dataset object to represent data
  with named fields (:ghpull:`301`).
- The :class:`~.JUNGFRAU` data access class now recognises 'JF500K' in source
  names (:ghpull:`300`).
- Fix sending around FileAccess objects with cloudpickle, which is used by Dask
  and clusterfutures (:ghpull:`303`).
- Fix permissions errors from opening the run files map JSON files
  (:ghpull:`304`).
- Fix errors opening runs with ``data='all'`` with an empty proc folder
  (:ghpull:`317`).
- The ``QuickView`` class deprecated in version 1.9 was removed.

1.11
----

2022-03-21

- New ``keep_dims`` option for :meth:`~.DataCollection.trains`,
  :meth:`~.DataCollection.train_from_id` and :meth:`~.DataCollection.train_from_index`.
  Normally the trains/pulses dimension is dropped from the arrays these methods
  return if it has length 1, but passing ``keep_dims=True`` will preserve this
  dimension (:ghpull:`288`).
- New :meth:`~.LPD1M.select_trains` and :meth:`~.LPD1M.split_trains` methods
  for multi-module detector data (:ghpull:`278`).
- :meth:`~.DataCollection.select` now accepts a list of source name patterns,
  which is more convenient for some use cases (:ghpull:`287`).
- Fix ``open_run(..., data='all')`` for runs with no proc data (:ghpull:`281`).
- Fix single run status when opening a run with a virtual overview file
  (:ghpull:`290`).
- Sources with no data recorded in a run are now represented in virtual overview
  files (:ghpull:`287`).
- Fix a race condition where files were closed in one thread as they were opened
  in another (:ghpull:`289`).


1.10
----

2022-02-01

- EXtra-data can now generate and use "virtual overview" files (:ghpull:`69`).
  A virtual overview file is a single file containing the metadata and indices
  of an entire run, and links to the source files for the data (using HDF5
  virtual datasets). When virtual overview files are available, :func:`open_run`
  and :func:`RunDirectory` will use them automatically; this should make it
  faster to open and explore runs (but not to read data).
- You can now specify ``parallelize=False`` for :func:`open_run` and
  :func:`RunDirectory` to open files in serial (:ghpull:`158`). This can be
  necessary if you're opening runs inside a parallel worker.
- Fix various features to work when 0 trains of data are selected (:ghpull:`260`).
- Fix :meth:`~.DataCollection.union` when starting with already-unioned data
  from different runs (:ghpull:`261`).
- Fix for opening runs with ``data='all'`` and combining data in certain ways
  (:ghpull:`274`).
- Fixes to ensure that files are not unnecessarily reopened (:ghpull:`264`).

1.9.1
-----

2021-11-30

- Fix errors from :meth:`~.KeyData.data_counts` and
  :meth:`~.KeyData.drop_empty_trains` when different train IDs exist for
  different sources (:ghpull:`257`).

1.9
---

2021-11-25

- New :meth:`.KeyData.as_single_value` method to check that a key remains
  constant (within a specified tolerance) through the data, and return it as
  a single value (:ghpull:`228`).
- New :meth:`.KeyData.train_id_coordinates` method to get train IDs associated
  with specific data as a NumPy array (:ghpull:`226`).
- :ref:`cmd-validate` now checks that timestamps in control data are in
  increasing order (:ghpull:`94`).
- Ensure basic :class:`DataCollection` functionality, including getting values
  from ``RUN`` and inspecting the shape & dtype of other data, works when no
  trains are selected (:ghpull:`244`).
- Fix reading data where some files in a run contain zero trains, as seen in
  some of the oldest EuXFEL data (:ghpull:`225`).
- Minor performance improvements for :meth:`~.DataCollection.select` when
  selecting single keys (no wildcards) and when selecting all keys along with
  ``require_all=True`` (:ghpull:`248`).

Deprecations & potentially breaking changes:

- The ``QuickView`` class is deprecated. We believe no-one is using this.
  If you are, please get in touch with da-support@xfel.eu .
- Removed the ``h5index`` module and the ``hdf5_paths`` function, which were
  deprecated in 1.7.

1.8.1
-----

2021-11-01

- Fixed two different bugs introduced in 1.8 affecting loading data for
  multi-module detectors with :meth:`~.LPD1M.get_array` when only some of the
  modules captured data for a given train (:ghpull:`234`).
- Fix ``open_run(..., data='all')`` when all sources in the raw data are copied
  to the corrected run folder (:ghpull:`236`).

1.8
---

2021-10-06

- New API for inspecting the data associated with a single source (:ghpull:`206`).
  Use a source name to get a :class:`.SourceData` object::

    xgm = run['SPB_XTD9_XGM/DOOCS/MAIN']
    xgm.keys()  # List the available keys
    beam_x = xgm['beamPosition.ixPos'].ndarray()

  See :ref:`data-by-source-and-key` for more details.
- Combining data from the same run with :meth:`~.union` now preserves
  'single run' status, so :meth:`~.run_metadata` still works (:ghpull:`208`).
  This only works with more recent data (file format version 1.0 and above).
- Reading data for multi-module detectors with :meth:`~.LPD1M.get_array` is
  now faster, especially when selecting a subset of pulses (:ghpull:`218`,
  :ghpull:`220`).
- Fix :meth:`~.data_counts` when data is missing for some selected trains
  (:ghpull:`222`).

Deprecations & potentially breaking changes:

- The ``numpy_to_cbf`` and ``hdf5_to_cbf`` functions have been removed
  (:ghpull:`213`), after they were deprecated in 1.7. If you need to create CBF
  files, consult the `Fabio package <http://www.silx.org/doc/fabio/latest/>`_.
- Some packages required for :ref:`cmd-serve-files` are no longer installed
  along with EXtra-data by default (:ghpull:`211`). Install with
  ``pip install extra-data[bridge]`` if you need this functionality.

1.7
---

2021-08-03

- New methods to split data into chunks with a similar number of trains in
  each: :meth:`.DataCollection.split_trains` and :meth:`.KeyData.split_trains`
  (:ghpull:`184`).
- New method :meth:`.KeyData.drop_empty_trains` to select only trains with
  data for a given key (:ghpull:`193`).
- Virtual CXI files can now be made for multi-module JUNGFRAU detectors
  (:ghpull:`62`).
- ``extra-data-validate`` now checks INDEX for control sources as well as
  instrument sources (:ghpull:`188`).
- Fix opening some files written by a test version of the DAQ, marked with
  format version 1.1 (:ghpull:`198`).
- Fix making virtual CXI files with h5py 3.3 (:ghpull:`195`).

Deprecations & potentially breaking changes:

- Remove special behaviour for :meth:`~.get_series` with big detector data,
  deprecated in 1.4 (:ghpull:`196`).
- Deprecated some functions for converting data to CBF format, and the
  ``h5index`` module (:ghpull:`197`). We believe these were unused.


1.6.1
-----

2021-05-14

- Fix a check which made it very slow to open runs with thousands of files
  (:ghpull:`183`).

1.6
---

2021-05-11

- :ref:`suspect-trains` are now included by default (:ghpull:`178`). Pass
  ``inc_suspect_trains=False`` to exclude them (as in 1.5), or the
  ``--exc-suspect-trains`` option for :ref:`cmd-make-virtual-cxi`.
- :func:`.open_run` can now combine raw & proc data when called with
  ``data='all'`` (:ghpull:`174`).
- Several new methods for accessing different kinds of metadata:

  - :meth:`.DataCollection.run_metadata` - per-run metadata including timestamps
    and proposal number (:ghpull:`175`)
  - :meth:`.DataCollection.get_run_value` and
    :meth:`.DataCollection.get_run_values` - per-run data from the control
    system (:ghpull:`164`)

- Selecting pulses should work for :meth:`.LPD1M.get_array` in parallel gain
  mode (:ghpull:`173`)
- Several fixes for handling 'suspect' train IDs (:ghpull:`172`).
- h5py >= 2.10 is now required (:ghpull:`177`).

1.5
---

2021-04-22

- Exclude :ref:`suspect-trains`, fixing occasional issues in particular with
  AGIPD data containing bad train IDs (:ghpull:`121`).
- Avoid converting train IDs to floats when using
  ``run.select(..., require_all=True)`` (:ghpull:`159`).
- New method :meth:`.train_timestamps` to get approximate timestamps for each
  train in the data (:ghpull:`165`)
- Checking whether a given source & key is present is now much faster in some
  cases (:ghpull:`170`).
- :ref:`cmd-lsxfel` can display structured datatypes nicely (:ghpull:`160`).
- :ref:`cmd-serve-files` can now send data on any ZMQ endpoint,
  not only ``tcp://`` sockets (:ghpull:`169`).
- Ensure :ref:`virtual CXI files <cmd-make-virtual-cxi>` created with EXtra-data
  can be read using HDF5 1.10 (:ghpull:`171`).
- Some fixes to make the test suite more robust (:ghpull:`156`, :ghpull:`167`,
  :ghpull:`169`).

1.4.1
-----

2021-03-10

- Fix :meth:`~.LPD1M.get_array` for raw DSSC & LPD data with multiple sequence
  files per module (:ghpull:`155`).
- Drop unnecessary dependency on scipy (:ghpull:`147`).

1.4
---

2021-02-12

New features:

- :meth:`.select` has a new option ``require_all=True`` to include only trains
  where all the selected sources & keys have data (:ghpull:`113`).
- :meth:`.select` now accepts :class:`DataCollection` and :class:`KeyData`
  objects, making it easy to re-select the same sources in another run
  (:ghpull:`114`).
- New classes for accessing data from :class:`.AGIPD500K` and :class:`.JUNGFRAU`
  multi-module detectors (:ghpull:`139`, :ghpull:`140`).
- New options for :func:`.stack_detector_data` to allow it to work with
  different data formats, including JUNGFRAU detectors (:ghpull:`141`).
- New option for :class:`.LPD1M` to read data taken in 'parallel gain' mode,
  giving it useful axis labels (:ghpull:`122`).
- :meth:`~.LPD1M.get_array` for multi-module detectors has a new option to label
  frames with memory cell IDs instead of pulse IDs (:ghpull:`101`).
- :meth:`.DataCollection.trains` can now optionally yield flat, single level
  dictionaries with ``(source, key)`` keys instead of nested dictionaries
  (:ghpull:`112`).
- New method :meth:`.KeyData.data_counts` (:ghpull:`92`).
- Labelled arrays from :meth:`.KeyData.xarray` and
  :meth:`.DataCollection.get_array` now have a name made from the source & key
  names, or as specified by the ``name=`` parameter (:ghpull:`87`).

Deprecations & potentially breaking changes:

- Earlier versions of EXtra-data unintentionally converted integer data from
  multi-module detectors to floats (in :meth:`~.LPD1M.get_array` and
  :meth:`~.LPD1M.get_dask_array`) with the special value NaN for missing data.
  This version preserves the data type, but missing integer data will be filled
  with 0. If this is not suitable, you can use the ``min_modules`` parameter
  to get only trains where all modules have data, or pass
  ``astype=np.float64, fill_value=np.nan`` to convert data to floats and fill
  gaps with NaN as before.
- Special handling in :meth:`~.get_series` to label some fast detector data with
  pulse IDs was deprecated (:ghpull:`131`). We believe no-one is using this.
  If you are, please contact da-support@xfel.eu to discuss alternatives.

Fixes and improvements

- Prevent :meth:`~.select` from rediscovering things that had previously been
  excluded from the selection (:ghpull:`128`).
- Fix default fill value for uint64 data in :func:`stack_detector_data`
  (:ghpull:`103`).
- Don't convert integer data to floats in :meth:`~.LPD1M.get_array` and
  :meth:`~.LPD1M.get_dask_array` methods for multi-module detector data
  (:ghpull:`98`).
- Documented the :class:`.KeyData` interface added in 1.3 (:ghpull:`96`)
- Fix ``extra-data-validate`` when a file cannot be opened (:ghpull:`93`).
- Fix name of ``extra-data-validate`` in its own help info (:ghpull:`90`).

1.3
---

2020-08-03

New features:

.. This directive allows the :option: below to link correctly.
.. program:: extra-data-make-virtual-cxi

- A new interface for data from a single source & key: use ``run[source, key]``
  to get a ``KeyData`` object, which can inspect and load the data from
  several sequence files (:ghpull:`70`).
- Methods which took a ``by_index`` object now accept slices (e.g.
  ``numpy.s_[:10]``) or indices directly (:ghpull:`68`, :ghpull:`79`). This
  includes :meth:`~.DataCollection.select_trains`,
  :meth:`~.DataCollection.get_array` and various methods for multi-module
  detectors, described in :doc:`agipd_lpd_data`.
- ``extra-data-make-virtual-cxi`` :option:`--fill-value` now accepts numbers in
  hexadecimal, octal & binary formats, e.g. ``0xfe`` (:ghpull:`73`).
- Added an ``unstack`` parameter to the :meth:`~.LPD1M.get_array` method for
  multi-module detectors, making it possible to retrieve an array as the data
  is stored, without separating the train & pulse axes (:ghpull:`72`).
- Added a ``require_all`` parameter to the :meth:`~.LPD1M.trains` method for
  multi-module detectors, to allow iterating with incomplete frames included
  (:ghpull:`77`).
- New :func:`.identify_multimod_detectors` function to find multi-module
  detectors in the data (:ghpull:`61`).

Fixes and improvements:

- Fix writing selected detector frames with :meth:`~.LPD1M.write_frames`
  for corrected data (:ghpull:`82`).
- Fix compatibility with pandas 1.1 (:ghpull:`83`).
- The :meth:`~.DataCollection.trains` iterator no longer includes zero-length
  arrays when a source has no data for that train (:ghpull:`75`).
- Fix a test which failed when run as root (:ghpull:`67`).

1.2
---

2020-06-04

New features:

- New :option:`karabo-bridge-serve-files --append-detector-modules` option
  to combine data from multiple detector modules. This makes streaming large
  detector data more similar to the live data streams (:ghpull:`40` and
  :ghpull:`51`).
- :ref:`cmd-serve-files` has new options to control the ZMQ socket and the use
  of an infiniband network interface (:ghpull:`50`). It also works with
  newer versions of the ``karabo_bridge`` Python package.
- New options to filter files from dCache which are unavailable or need to be
  read from tape when opening a run (:ghpull:`35`). This also comes with a new
  command :ref:`cmd-locality` to inspect this information.
- New :option:`lsxfel --detail` option to show more detail on selected sources
  (:ghpull:`38`).
- New :option:`extra-data-make-virtual-cxi --fill-value` option to control the
  fill value for missing data (:ghpull:`59`)
- New method :meth:`~.LPD1M.write_frames` to save a subset of detector frames
  to a new file in EuXFEL HDF5 format (:ghpull:`47`).
- :meth:`DataCollection.select` can take arbitrary iterables of patterns,
  rather than just lists (:ghpull:`43`).

Fixes and improvements:

- EXtra-data now tries to manage how many HDF5 files it has open at one time,
  to avoid hitting a limit on the total number of open files in a process
  (:ghpull:`25` and :ghpull:`48`).
  Importing EXtra-data will now raise this limit as far as it can (to 4096
  on Maxwell), and try to keep the files it handles to no more than half of
  this. Files should be silently closed and reopened as needed, so this
  shouldn't affect how you use it.
- A better way of creating Dask arrays to avoid problems with Dask's local
  schedulers, and with arrays comprising very large numbers of files
  (:ghpull:`63`).
- The classes for accessing multi-module detector data (see
  :doc:`agipd_lpd_data`) and writing virtual CXI files no longer assume that
  the same number of frames are recorded in every train (:ghpull:`44`).
- Fix validation where a file has no trains at all (:ghpull:`42`).
- More testing of EuXFEL file format version 1.0 (:ghpull:`56`).
- Test coverage measurement fixed with multiprocessing (:ghpull:`37`).
- Tests switched from ``mock`` module to ``unittest.mock`` (:ghpull:`52`).

1.1
---

2020-03-06

- Opening and validating run directories now handles files in parallel, which
  should make it substantially faster (:ghpull:`30`).
- Various data access operations no longer require finding all the keys for
  a given data source, which saves time in certain situations (:ghpull:`24`).
- :func:`~.open_run` now accepts numpy integers for proposal and run numbers,
  as well as standard Python integers (:ghpull:`34`).
- :ref:`Run map cache files <run-map-caching>` can be saved on the EuXFEL online
  cluster, which speeds up reopening runs there (:ghpull:`36`).
- Added tests with simulated bad files for the validation code (:ghpull:`23`).

1.0
---

2020-02-21

- New :meth:`~.LPD1M.get_dask_array` method for accessing detector data with
  Dask (:ghpull:`18`).
- Fix ``extra-data-validate`` with a run directory without a :ref:`cached data
  map <run-map-caching>` (:ghpull:`12`).
- Add ``.squeeze()`` method for virtual stacks of detector data from
  :func:`.stack_detector_data` (:ghpull:`16`).
- Close each file after reading its metadata, to avoid hitting the limit of
  open files when opening a large run (:ghpull:`8`).
  This is a mitigation: you will still hit the limit if you access data from
  enough files. The default limit on Maxwell is 1024 files, but you can
  raise this to 4096 using the Python
  `resource module <https://docs.python.org/3/library/resource.html>`_.
- Display progress information while validating a run directory (:ghpull:`19`).
- Display run duration to only one decimal place (:ghpull:`5`).
- Documentation reorganised to emphasise tutorials and examples (:ghpull:`10`).

This version requires Python 3.6 or above.

0.8
---

2019-11-18

First separated version. No functional changes from karabo_data 0.7.

Earlier history
---------------

The code in EXtra-data was previously released as *karabo_data*, up to version
0.7. See the `karabo_data release notes
<https://karabo-data.readthedocs.io/en/latest/changelog.html>`_ for changes
before the renaming.
