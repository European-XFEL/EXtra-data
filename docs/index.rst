EXtra-data
==========

**EXtra-data** is a Python library for accessing saved data
produced at `European XFEL <https://www.xfel.eu/>`_.

Installation
------------

EXtra-data is available in our Python environment on the Maxwell cluster::

    module load exfel exfel-python

You can also install it `from PyPI <https://pypi.org/project/EXtra-data/>`__
to use in other environments with Python 3::

    pip install extra_data

This will install the `extra_data` package and the most commonly useful dependencies.
Some large dependencies or dependencies only required for specific functionalities are
not installed by default. You can use pip to install everything required for extra or
all uses of `extra_data` (e.g. Dask Array, :ref:`cmd-serve-files`). This installs both
`extra_data` and dependencies that are necessary::

    pip install "extra_data[bridge]"    # install dependencies for karabo-bridge-like data streaming
    pip install "extra_data[complete]"  # install dependencies for all features

If you get a permissions error, add the ``--user`` flag to that command.

Quickstart
----------

Open a run on the Maxwell cluster::

    from extra_data import open_run

    run = open_run(proposal=700000, run=1)

You can also specify a run directory, or open an individual file - see
:ref:`opening-files` for details. The same methods to access data work with any
of these options.

Load data as a NumPy array for a given source & key::

    arr = run["SA3_XTD10_PES/ADC/1:network", "digitizers.channel_4_A.raw.samples"].ndarray()

You can load only a region of interest, get a labelled array with train IDs,
or load 1D data as columns in a pandas dataframe. See :doc:`xpd_examples`
(example) and :ref:`data-by-source-and-key` (reference) for more information.

For data that's too big to fit in memory at once, you can read one pulse train
at a time::

    for train_id, data in run.select("*/DET/*", "image.data").trains():
        mod0 = data["FXE_DET_LPD1M-1/DET/0CH0:xtdf"]["image.data"]

Other options to work with large data volumes include breaking the run into
smaller parts with :meth:`~.DataCollection.split_trains` before loading data,
and automatic chunking with the `Dask <https://dask.org/>`_ framework and
:meth:`~.dask_array`.

Documentation contents
----------------------

.. toctree::
   :caption: Tutorials and Examples
   :maxdepth: 2

   xpd_examples
   inspection
   iterate_trains
   aligning_trains
   dask_averaging
   parallel_example
   lpd_data
   xpd_examples2
   

.. toctree::
   :caption: Reference
   :maxdepth: 2

   reading_files
   agipd_lpd_data
   streaming
   validation
   cli
   data_format
   performance

.. toctree::
   :caption: Development
   :maxdepth: 1

   changelog
   architecture

.. seealso::

   `Data Analysis at European XFEL
   <https://rtd.xfel.eu/docs/data-analysis-user-documentation/en/latest/>`_

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

