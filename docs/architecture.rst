Architecture
============

.. note::

   This page describes technical details about EXtra-data. You shouldn't need
   this information to use it.

Objects
-------

There are three classes making up the core API of EXtra-data:

- :class:`.DataCollection` is what you get from :ref:`opening a run or file
  <opening-files>`: data for several sources over some range of pulse trains
  (i.e. time). It has methods to :ref:`select a subset of that data
  <selecting-combining>`.
- :class:`.SourceData` comes from ``run[source]``, representing one source, such
  as a motor or a detector module. Each source has a set of keys.
- :class:`.KeyData` comes from ``run[source, key]``, representing data for a
  single source & key. This has a dtype and a shape like a NumPy array, but
  the data is not in memory. It has methods to load the data as a NumPy array,
  an Xarray DataArray, or a Dask array.

Component classes for :doc:`multi-module detectors <agipd_lpd_data>` build on
top of this core to work more conveniently with major data sources. There are
more component classes in the `EXtra package <https://extra.readthedocs.io/en/latest/>`_.

:class:`.FileAccess` is a lower-level class to manage access to a single
:doc:`EuXFEL format HDF5 file <data_format>`, including caching index information.
There should only be one ``FileAccess`` object per file on disk, even if
multiple ``DataCollection``, ``SourceData`` and ``KeyData`` objects refer to it.

Modules
-------

- ``cli`` contains command-line interfaces.
- ``components`` provides interfaces that bring together data from several
  similar sources, i.e. multi-module detectors where each module is a separate
  source.
- ``exceptions`` defines some custom error classes.
- ``export`` sends data from files over ZMQ in the Karabo Bridge format.
- ``file_access`` contains :class:`.FileAccess` (described above), along with
  machinery to keep the number of open files under a limit.
- ``keydata`` contains :class:`.KeyData` (described above).
- ``locality`` can check whether files are available on disk or on tape
  in a `dCache <https://www.dcache.org/>`_ filesystem.
- ``lsxfel`` is the :ref:`cmd-lsxfel` command.
- ``reader`` contains :class:`.DataCollection` (described above), and functions
  to open a run or a file.
- ``read_machinery`` is a collection of pieces that support ``reader``.
- ``run_files_map`` manages caching metadata about the files of a run in a
  JSON file, to speed up opening the run.
- ``sourcedata`` contains :class:`.SourceData` (described above).
- ``stacking`` has functions for stacking multiple arrays into one, another
  option for working with multi-module detector data.
- ``utils`` is miscellaneous pieces that don't fit anywhere else.
- ``validation`` checks if files & runs have the expected format, for the
  :ref:`cmd-validate` command.
- ``writer`` writes data in EuXFEL format files, for
  :meth:`~.DataCollection.write` and :meth:`~.DataCollection.write_virtual`.
- ``write_cxi`` makes CXI format HDF5 files using virtual datasets to
  expose multi-module detector data. Used by :meth:`~.LPD1M.write_virtual_cxi`
  and the :ref:`cmd-make-virtual-cxi` command.
