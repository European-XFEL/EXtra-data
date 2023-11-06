Streaming data over ZeroMQ
==========================

*Karabo Bridge* provides access to live data during the experiment over a ZeroMQ
socket. The ``extra_data`` Python package can stream data from files using the same
protocol. You can use this to test code which expects to receive data from
Karabo Bridge, or use the same code for analysing live data and stored data.

To stream data from a saved run, use the ``karabo-bridge-serve-run`` command:

.. code-block:: shell

   #                     Proposal run
   karabo-bridge-serve-run 700000 40 --port 4545 \
        --include 'SPB_IRDA_JF4M/DET/JNGFR*:daqOutput' \
        --include '*/MOTOR/*[*Position]'

The port number (4545 above) must be an unused TCP port above 1024.
Clients will need this port and the IP address of the sender to connect.
For clients running on the same node, use the IP address ``127.0.0.1``.
Command-line options are explained on the
:ref:`command reference <cmd-serve-run>` page.

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
