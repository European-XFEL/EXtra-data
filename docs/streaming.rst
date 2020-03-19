Streaming data over ZeroMQ
==========================

*Karabo Bridge* provides access to live data during the experiment over a ZeroMQ
socket. The ``extra_data`` Python package can stream data from files using the same
protocol. You can use this to test code which expects to receive data from
Karabo Bridge, or use the same code for analysing live data and stored data.

To stream the data from a file or run unmodified, use the command::

    karabo-bridge-serve-files /gpfs/exfel/exp/SPB/201830/p900022/raw/r0034 4545

The number (4545) must be an unused TCP port above 1024. It will bind to
this and stream the data to any connected clients.

Keyword-argument options:

``--source SOURCE``: only sources matching the string SOURCE will be streamed.
Default is '*' as a global wildcard for all sources.

``--key KEY``: only data sets keyed by the string KEY will be streamed.
Default is '*' as a global wildcard for all keys.   

``--append-detector-modules``: given that the data read from file contains
multiple detector modules as separate sources, which is the case for big area
detectors like AGIPD, LPD and DSSC, append these into one single source, as
expected by certain software (e.g. OnDA).

The last optional feature has only been tested for AGIPD-1M data (so far).
One should use this only for data runs that actually contain HDF5 files from
AGIPD sources, **and** make a selection like ``--source "*/DET/*"``, because a
global selection of all sources will cause an error if additional non-detector
sources are found.

We provide Karabo bridge clients as Python and C++ libraries.

If you want to do some processing on the data before streaming it, you can
use this Python interface to send it out:

.. module:: extra_data.export

.. autoclass:: ZMQStreamer

   .. automethod:: start

   .. automethod:: feed
