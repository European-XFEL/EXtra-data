Command line tools
==================

.. _cmd-lsxfel:

``lsxfel``
----------

Examine the contents of an EuXFEL proposal directory, run directory, or HDF5
file:

.. code-block:: shell

   # Proposal directory
   lsxfel /gpfs/exfel/exp/XMPL/201750/p700000

   # Run directory
   lsxfel /gpfs/exfel/exp/XMPL/201750/p700000/raw/r0002

   # Single file
   lsxfel /gpfs/exfel/exp/XMPL/201750/p700000/proc/r0002/CORR-R0034-AGIPD00-S00000.h5

.. program:: lsxfel

.. option:: --detail <source-pattern>

   Show more detail on the keys and data of the sources selected by a pattern
   like ``*/XGM/*``. Only applies when inspecting a single run or file.
   Can be used several times to select different patterns.

   This option can make ``lsxfel`` considerably slower.

.. _cmd-validate:

``extra-data-validate``
------------------------

Check the structure of an EuXFEL run or HDF5 file:

.. code-block:: shell

   extra-data-validate /gpfs/exfel/exp/XMPL/201750/p700000/raw/r0002

If it finds problems with the data, the program will produce a list of them and
exit with status 1. See :doc:`validation` for details of what it checks.

.. _cmd-serve-files:

``karabo-bridge-serve-files``
-----------------------------

Stream data from files in the `Karabo bridge
<https://rtd.xfel.eu/docs/data-analysis-user-documentation/en/latest/online.html#streaming-from-karabo-bridge>`_
format. See :doc:`streaming` for more information.

.. code-block:: shell

   karabo-bridge-serve-files /gpfs/exfel/exp/XMPL/201750/p700000/proc/r0005 4321

.. program:: karabo-bridge-serve-files

.. option:: --source <source>

   Only sources matching the string <source> will be streamed. Default is '*',
   serving as a global wildcard for all sources.

.. option:: --key <key>

   Only data sets keyed by the string <key> will be streamed. Default is '*',
   serving as a global wildcard for all keys.

.. option:: --append-detector-modules

   If the file data contains multiple detector modules as separate sources,
   i. e. for big area detectors (AGIPD, LPD and DSSC), append these into one
   single source.

.. option:: --dummy-timestamps

   Add mock timestamps if missing in the original meta-data.

These two options above - appended module sources and dummy timestamps - are
required if streamed data shall be provided to OnDA.

.. option:: -z <type>, --socket-type <type>

   The ZMQ socket type to use, one of ``PUB``, ``PUSH`` or ``REP``.
   Default: ``REP``.

.. option:: --use-infiniband

   Use the infiniband network interface (``ib0``) if it's present.

.. _cmd-make-virtual-cxi:

``extra-data-make-virtual-cxi``
--------------------------------

Make a virtual CXI file to access AGIPD/LPD/JUNGFRAU detector data from a specified run:

.. code-block:: shell

   extra-data-make-virtual-cxi /gpfs/exfel/exp/XMPL/201750/p700000/proc/r0003 -o xmpl-3.cxi

.. program:: extra-data-make-virtual-cxi

.. option:: -o <path>, --output <path>

   The filename to write. Defaults to creating a file in the proposal's
   scratch directory.

.. option:: --min-modules <number>

   Include trains where at least N modules have data (default: half+1 of all detector modules).

.. option:: --n-modules <number>

   Number of detector modules in the experiment setup. Should be used only for JUNGFRAU data.

.. option:: --fill-value <dataset> <value>

   Set the fill value for dataset (one of ``data``, ``gain`` or ``mask``).
   The defaults are different in different cases:

   - data (raw, uint16): 0
   - data (proc, float32): NaN
   - gain: 0
   - mask: 0xffffffff

.. option:: --exc-suspect-trains

   Exclude :ref:`suspect-trains` from the data to assemble. This can fix some
   problems with bad train IDs.

.. _cmd-locality:

``extra-data-locality``
------------------------

Check how the files are stored:

.. code-block:: shell

   extra-data-locality /gpfs/exfel/exp/XMPL/201750/p700000/raw/r0002

The file reading may hang for a long time if files are unavailable or require staging
in dCache from the tape. The program helps finding problem files.

If it finds problems with the data locality, the program will produce a list of files
located on tape, lost or at unknown locality and exit with the non-zero status.
