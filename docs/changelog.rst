Release Notes
=============

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
