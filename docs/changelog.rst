Release Notes
=============

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
