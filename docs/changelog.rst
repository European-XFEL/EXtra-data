Release Notes
=============

.. _rel-1.23.2:

1.23.2
------

2025-12-09

- Fix a bug with the new per-instrument default aliases files, which are not
  currently accessible for external users (:ghpull:`694`).

.. _rel-1.23.1:

1.23.1
------

2025-12-08

- Internal fix for testing mechanisms.

.. _rel-1.23:

1.23
----

2025-12-04

- Aliases can now be set in a per-instrument default file at
  ``/gpfs/exfel/sw/<instrument>/extra-data-aliases-default.yml`` (:ghpull:`682`).
  This file is copied to proposal folders the first time a run is opened, so
  changing the defaults only affects new proposals.
- Coordinate labels can now be specified for :meth:`.KeyData.xarray` with the
  ``extra_coords`` parameter (:ghpull:`672`). Also, if dimensions are named
  using ``extra_dims``, default integer coordinate labels will be added to any
  dimensions not given in ``extra_coords``.
- Support for reading the ERRATA & REDUCTION sections of EXDF files through
  ``run.auxiliary`` (:ghpull:`651`).
- New command ``extra-data-readable`` for checking if HDF5 data files can be
  read without hanging (:ghpull:`675`).
- Fix finding an arbitrary key for control sources with keys selected
  (:ghpull:`648`).
- Fix getting information from filenames when using a virtual overview file
  (:ghpull:`652`).
- Speed up getting information from filenames when not using a virtual overview
  file (:ghpull:`688`).

.. _rel-1.22:

1.22
----

2025-07-16

- Train selection now accepts boolean arrays, either Numpy arrays of suitable
  length or xarray arrays with a ``trainId`` coordinate (:ghpull:`612`).
- The ``run.alias`` repr now includes links to edit any alias files in use, via
  max-jhub (:ghpull:`640`).
- The :option:`lsxfel --detail` option will do a substring match if the argument
  doesn't look like a glob pattern (containing ``*?[``) (:ghpull:`628`).
- New :option:`lsxfel --aggregators` option to show the data aggregator saving
  each source (:ghpull:`625`).
- Reading compressed detector data in parallel (added in :ref:`rel-1.21`) can
  now work around 15% faster (:ghpull:`633`)
- Fix tab completion on aliases (:ghpull:`626`).
- Fix :meth:`~.as_single_value` with string data (:ghpull:`623`).
- Fix handling of CONTROL sources with only RUN keys (:ghpull:`622`).
- Fix streaming karabo-bridge data from files including control data with arrays
  of strings (:ghpull:`616`).
- :func:`~.open_run` accepts a path-like object for ``proposal=`` (:ghpull:`641`).
- Make ``zlib_into`` an optional dependency rather than absolutely required
  (:ghpull:`621`).
- Fix a rare race condition when creating a virtual overview file, which could
  allow something to read a partially written file (:ghpull:`624`).
- Support for custom error messages in :exc:`SourceNameError` (:ghpull:`619`).

Breaking changes
~~~~~~~~~~~~~~~~

- String data stored from control sources is now returned as ``str`` objects
  rather than ``bytes`` from various methods (:ghpull:`623`).


.. _rel-1.21:

1.21
----

2025-03-24

- Detector data classes such as :class:`~.AGIPD1M` can now decompress data in
  parallel, providing a significant speedup for reading compressed data
  (:ghpull:`593`). This is used by default with 16 threads on suitable data,
  and can be controlled by passing ``decompress_threads=N`` to ``.ndarray()``
  or ``.xarray()`` methods, or setting the ``EXTRA_NUM_THREADS`` environment
  variable. Specify 1 thread to use HDF5's single-threaded decompression.
- The ``.pulse_id_coordinates()`` and ``.cell_id_coordinates()`` methods on
  AGIPD, DSSC & LPD data objects now respect pulse selections (:ghpull:`604`).
- Fix running the :ref:`cmd-validate` command with no ``--skip`` parameter
  (:ghpull:`606`).

.. _rel-1.20:

1.20
----

2025-02-26

- Loading data as an xarray object will now include the units symbol as a
  attribute called ``units`` (:ghpull:`592`).
- Some improvements to virtual overview files when one sequence file is missing
  data (:ghpull:`600`) or when no data was recorded for a particular source in
  an entire run (:ghpull:`601`, :ghpull:`602`).
- EXtra-data now requires Python 3.10 or above (:ghpull:`294`).

.. _rel-1.19:

1.19
----

2025-01-24

- :func:`~.open_run` now combines raw & corrected data by default, preferring
  raw for source names found in both (:ghpull:`569`). This means corrected
  detector data is visible by default in recent runs.
- :doc:`Detector data classes <agipd_lpd_data>` can now select corrected or raw
  data with the parameter ``raw=False`` or ``True`` (:ghpull:`558`). If this is
  not specified, they will use corrected data if available, and raw if not, in
  line with the previous behaviour. This also depends on how you open the run.
- ``source_name in run`` and ``(source_name, key_name) in run`` now work
  (:ghpull:`582`).
- You can now select train IDs in DataCollection and SourceData like
  ``run[tids]`` (:ghpull:`559`)
- Make it easier to select a single train ID using ``by_id``, and fix raising
  IndexError when selecting a single train index as an integer (:ghpull:`558`).
- You can use the ``|`` operator to combine multiple :class:`DataCollection`
  or :class:`SourceData` objects, equivalent to their
  :meth:`~.DataCollection.union` methods (:ghpull:`582`).
- New option ``run[source].run_values(inc_timestamps=False)`` to get a dict of
  run values excluding timestamps (:ghpull:`581`).
- Specific parts of :doc:`validation <validation>` can now be skipped with a new
  :option:`extra-data-validate --skip` option (:ghpull:`522`).
- Avoid memory errors & improve performance of reading XTDF detector data with
  a pulse selection (:ghpull:`576`).
- Fix ``det.masked_data().select_pulses()`` in XTDF detector components
  (:ghpull:`571`)
- Fix using ``file_filter`` parameter when opening a run (:ghpull:`566`)
- PyYAML is now a full dependency (:ghpull:`577`).

.. _rel-1.18:

1.18
----

2024-09-23

-  EXtra-data now requires Python 3.9 or above (:ghpull:`554`).
-  Aliases are now case-insensitive, and allow - & \_ interchangeably, so ``las-x``
   and ``Las_X`` are considered the same (:ghpull:`515`).
-  Add concept of 'legacy' source names, references to sources which have been
   renamed (:ghpull:`527`). This will be used for calibrated detector data.
-  Add source, key & alias completions for IPython (:ghpull:`514`).
-  New ``.masked_data()`` method to load detector data with mask (:ghpull:`518`).
   See :doc:`agipd_lpd_data`.
-  A new ``euxfel_local_time`` option for :meth:`.DataCollection.train_timestamps`
   to convert timestamps to local (German) time (:ghpull:`550`).
-  Return timezone-aware values from :meth:`~.DataCollection.train_timestamps`
   where possible (:ghpull:`550`).
-  Allow ``kd[trains]`` for multi-module KeyData objects (:ghpull:`520`).
-  Add optional index group filter to :meth:`.SourceData.one_key` (:ghpull:`526`).
-  Fixed various compatibility issues with Numpy 2.0 (:ghpull:`530`).
-  Allow caching file maps from 'open' & 'red' run folders in the proposal
   scratch folder (:ghpull:`548`, :ghpull:`549`).
-  When the file map is cached in multiple places, read the newest version
   (:ghpull:`524`).
-  Prevent unwanted iteration over a KeyData object (:ghpull:`519`).
-  Fix making virtual CXI files for JUNGFRAU data if the 'mask' dataset is not
   present (:ghpull:`511`).
-  Fix the message shown when skipping files because of how they're stored
   (:ghpull:`525`).

.. _rel-1.17:

1.17
----

2024-04-10

-  :func:`open_run` can now combine additional data locations besides the main
   raw & proc folders (:ghpull:`298`)::

       run = open_run(6616, 31, data=['raw', 'scratch/test_cal'])

   This specifies a list of paths under the proposal directory. The folders
   given should contain run folders with 4 digit run numbers, e.g. ``r0031``.
   If the same source names appear, those sources will be visible from the last
   location in the list.
-  Add ``.pulse_id_coordinates()`` & ``.train_id_coordinates()`` for XTDF image
   data (:ghpull:`506`).
-  Add :meth:`~.LPD1M.data_availability` method for multi-module detectors
   (:ghpull:`504`).
-  New ``include_empty`` option to include empty trains when iterating KeyData
   with :meth:`~.KeyData.trains` (:ghpull:`501`)
-  Support selecting down DataCollection by SourceData objects (:ghpull:`499`)
-  Merge attributes of key group and value dataset for CONTROL keys
   (:ghpull:`498`)
-  Add warning when :meth:`~.DataCollection.select` with ``require_all``
   discards all trains (:ghpull:`497`).
-  Miscellaneous improvements to ``.buffer_shape()`` method for multi-module
   detector data (:ghpull:`505`).
-  Return a copy of the array for ``detector_key.train_id_coordinates()``
   (:ghpull:`502`)

.. _rel-1.16:

1.16
----

2024-02-26

-  Fix loading aliases for old proposals (:ghpull:`490`).
-  Hide the message about proposal aliases when opening a run. (:ghpull:`478`).
-  ``extra-data-validate`` gives clearer messages for filesystem errors
   (:ghpull:`472`).
-  Fix OverflowError in lsxfel & run.info() with some corrupted train IDs
   (:ghpull:`489`).
-  Fix a selection of deprecation warnings (:ghpull:`469`).
-  Add a development tool to copy the structure of EuXFEL data files
   without the data (:ghpull:`467`).

.. _rel-1.15.1:

1.15.1
------

2023-11-17

- :class:`~.JUNGFRAU` recognises some additional naming patterns seen in new
  detector instances (:ghpull:`464`).

.. _rel-1.15:

1.15
----

2023-11-06

-  New properties :attr:`~.KeyData.units` and :attr:`~.KeyData.units_name` on
   ``KeyData`` objects to retrieve units metadata written by Karabo (:ghpull:`449`).
-  New command :ref:`cmd-serve-run` to more conveniently stream
   data from a saved run in Karabo Bridge format (:ghpull:`458`).
-  Fix :meth:`~.DataCollection.split_trains` being very slow when splitting a
   long run into  many pieces (:ghpull:`459`).
-  Include XTDF sources in :ref:`cmd-lsxfel` when details are enabled (:ghpull:`440`).

.. _rel-1.14:

1.14
----

2023-07-27

-  New ``train_id_coordinates`` method for source data, like the one for
   key data (:ghpull:`431`).
-  New attributes ``.nbytes``, ``.size_mb`` and ``.size_gb`` to
   conveniently see how much data is present for a given source & key
   (:ghpull:`430`).
-  Fix ``.ndarray(module_gaps=True)`` for xtdf detector data (:ghpull:`432`).

.. _rel-1.13:

1.13
----

2023-06-15

- Support for aliases (:ghpull:`367`), to provide shorter, more meaningful names
  for specific sources & keys, and support for loading a default set of aliases
  for the proposal when using :func:`~.open_run` (:ghpull:`398`). See
  :ref:`using-aliases` for more information.
- New APIs for multi-module detector data to work more like regular sources and
  keys, e.g. ``agipd['image.data'].ndarray()`` (:ghpull:`337`). These changes
  also change how Dask arrays are created for multi-module detector data,
  hopefully making them more efficient for typical use cases.
- New method :meth:`~.DataCollection.plot_missing_data` to show where sources
  are missing data for some trains (:ghpull:`402`).
- Merging data with :meth:`~.union` now applies the same train IDs to all
  included sources, whereas previously sources could have different train IDs
  selected (:ghpull:`416`).
- A new property ``run[src].device_class`` exposes the Karabo device class name
  for control sources (:ghpull:`390`).
- :class:`.JUNGFRAU` now accepts a ``first_modno`` for detectors where the first
  module is named with e.g. ``JNGFR03`` (:ghpull:`379`).
- ``run[src].is_control`` and ``.is_instrument`` properties (:ghpull:`403`).
- :class:`.SourceData` objects now have ``.data_counts()``,
  ``.drop_empty_trains()`` and ``.split_trains()`` methods like :class:`.KeyData`
  (:ghpull:`404`, :ghpull:`405`, :ghpull:`407`).
- New method ``SourceData.one_key()`` to quickly find an arbitrary key for a
  source.
- :meth:`~.DataCollection.select` now accepts a ``require_any=True`` parameter
  to filter trains where at least one of the selected sources & keys has data,
  complementing ``require_all`` (:ghpull:`400`).
- New property :attr:`KeyData.source_file_paths` to locate real data files even
  if the run was opened using a virtual overview file (:ghpull:`325`).
- New :class:`.SourceData` properties ``storage_class``, ``data_category`` and
  ``aggregator`` to extract details from the filename & folder path, for the
  main folder structure on EuXFEL compute clusters (:ghpull:`399`).
- It's now possible to ``pip install extra-data[complete]`` to install
  EXtra-data along with all optional dependencies (:ghpull:`414`).
- Fix for missing CONTROL data when
  :ref:`accessing data by train <data-by-train>` (:ghpull:`359`).
- Fix using ``with`` to open & close runs when a virtual overview file is found
  (:ghpull:`375`).
- Fix calling :func:`~.open_run` with ``data='all', parallelize=False``
  (:ghpull:`338`).
- Fix using :class:`.DataCollection` objects with multiprocessing and spawned
  subprocesses (:ghpull:`348`).
- Better error messages when files are missing ``INDEX`` or ``METADATA``
  sections (:ghpull:`361`).
- Fix creating virtual overview files with extended metadata when source files
  are format version 1.1 or newer (:ghpull:`332`).

.. _rel-1.12:

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

.. _rel-1.11:

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

.. _rel-1.10:

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

.. _rel-1.9.1:

1.9.1
-----

2021-11-30

- Fix errors from :meth:`~.KeyData.data_counts` and
  :meth:`~.KeyData.drop_empty_trains` when different train IDs exist for
  different sources (:ghpull:`257`).

.. _rel-1.9:

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

.. _rel-1.8.1:

1.8.1
-----

2021-11-01

- Fixed two different bugs introduced in 1.8 affecting loading data for
  multi-module detectors with :meth:`~.LPD1M.get_array` when only some of the
  modules captured data for a given train (:ghpull:`234`).
- Fix ``open_run(..., data='all')`` when all sources in the raw data are copied
  to the corrected run folder (:ghpull:`236`).

.. _rel-1.8:

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

.. _rel-1.7:

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

.. _rel-1.6.1:

1.6.1
-----

2021-05-14

- Fix a check which made it very slow to open runs with thousands of files
  (:ghpull:`183`).

.. _rel-1.6:

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

.. _rel-1.5:

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

.. _rel-1.4.1:

1.4.1
-----

2021-03-10

- Fix :meth:`~.LPD1M.get_array` for raw DSSC & LPD data with multiple sequence
  files per module (:ghpull:`155`).
- Drop unnecessary dependency on scipy (:ghpull:`147`).

.. _rel-1.4:

1.4
---

2021-02-12

New features:

- :meth:`~.DataCollection.select` has a new option ``require_all=True`` to include only trains
  where all the selected sources & keys have data (:ghpull:`113`).
- :meth:`~.DataCollection.select` now accepts :class:`DataCollection` and :class:`KeyData`
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

- Prevent :meth:`~.DataCollection.select` from rediscovering things that had previously been
  excluded from the selection (:ghpull:`128`).
- Fix default fill value for uint64 data in :func:`stack_detector_data`
  (:ghpull:`103`).
- Don't convert integer data to floats in :meth:`~.LPD1M.get_array` and
  :meth:`~.LPD1M.get_dask_array` methods for multi-module detector data
  (:ghpull:`98`).
- Documented the :class:`.KeyData` interface added in 1.3 (:ghpull:`96`)
- Fix ``extra-data-validate`` when a file cannot be opened (:ghpull:`93`).
- Fix name of ``extra-data-validate`` in its own help info (:ghpull:`90`).

.. _rel-1.3:

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

.. _rel-1.2:

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

.. _rel-1.1:

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

.. _rel-1.0:

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

.. _rel-0.8:

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
