Release Notes
=============

1.5
---

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

- Fix :meth:`~.LPD1M.get_array` for raw DSSC & LPD data with multiple sequence
  files per module (:ghpull:`155`).
- Drop unnecessary dependency on scipy (:ghpull:`147`).

1.4
---

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

First separated version. No functional changes from karabo_data 0.7.

Earlier history
---------------

The code in EXtra-data was previously released as *karabo_data*, up to version
0.7. See the `karabo_data release notes
<https://karabo-data.readthedocs.io/en/latest/changelog.html>`_ for changes
before the renaming.
