Architecture
============

.. note::

   This page describes technical details about EXtra-data. You shouldn't need
   this information to use it.

Objects
-------

The :class:`.DataCollection` class is the central piece of EXtra-data. It
represents a collection of XFEL data sources and their keys, for a set of train
IDs. It refers to data in one or more files (a run directory is often the
starting point). A subset of its sources/keys or train IDs may be selected to
make a new, more restricted :class:`.DataCollection`.

:class:`.KeyData` represents data for a single source & key, selected from a
``DataCollection`` like ``run[source, key]``. This data may still be spread
across several files. The data can be loaded into a NumPy array, among other
types.

:class:`.FileAccess` manages access to a single EuXFEL format HDF5 file,
including caching index information. There should only be one ``FileAccess``
object per file on disk, even if multiple ``DataCollection`` and ``KeyData``
objects refer to it.

Modules
-------

- ``cli`` contains command-line interfaces (so far only
  :ref:`cmd-make-virtual-cxi`).
- ``components`` provides interfaces that bring together data from several
  similar sources, i.e. multi-module detectors where each module is a separate
  source.
- ``exceptions`` defines some custom error classes.
- ``export`` sends data from files over ZMQ in the Karabo Bridge format.
- ``file_access`` contains :class:`.FileAccess` (described above), along with
  machinery to keep the number of open files under a limit.
- ``h5index`` lists datasets in an HDF5 file. Deprecated.
- ``keydata`` contains :class:`.KeyData` (described above).
- ``locality`` can check whether files are available on disk or on tape
  in a `dCache <https://www.dcache.org/>`_ filesystem.
- ``lsxfel`` is the :ref:`cmd-lsxfel` command.
- ``reader`` contains :class:`.DataCollection` (described above), and functions
  to open a run or a file.
- ``read_machinery`` is a collection of pieces that support ``reader``.
- ``run_files_map`` manages caching metadata about the files of a run in a
  JSON file, to speed up opening the run.
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
