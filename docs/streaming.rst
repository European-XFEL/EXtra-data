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
Command-line options are explained on the
:ref:`command reference <cmd-serve-files>` page.

.. note::

   If you install EXtra-data yourself, streaming data requires some optional
   dependencies. To ensure you have these, run::

      pip install extra_data[bridge]

   These dependencies are installed in the EuXFEL Anaconda installation on the
   Maxwell cluster.

We provide Karabo bridge clients as `Python
<https://github.com/European-XFEL/karabo-bridge-py>`__ and `C++ libraries
<https://github.com/European-XFEL/karabo-bridge-cpp>`__.

If you want to do some processing on the data before streaming it, you can
use this Python interface to send it out:

.. module:: extra_data.export

.. autoclass:: ZMQStreamer

   .. automethod:: start

   .. automethod:: feed
