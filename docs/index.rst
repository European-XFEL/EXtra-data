EXtra-data
==========

**EXtra-data** is a Python library for accessing and working with data
produced at `European XFEL <https://www.xfel.eu/>`_.

.. note::
   EXtra-data is the new name for karabo_data.
   The code to work with detector geometry has been separated as
   `EXtra-geom <https://github.com/European-XFEL/EXtra-geom>`__.

Installation
------------

EXtra-data is available on our Anaconda installation on the Maxwell cluster::

    module load exfel exfel_anaconda3

You can also install it `from PyPI <https://pypi.org/project/karabo-data/>`__
to use in other environments with Python 3.5 or later::

    pip install extra_data

If you get a permissions error, add the ``--user`` flag to that command.

Quickstart
----------

Open a run or a file - see :ref:`opening-files` for more::

    from extra_data import open_run, RunDirectory, H5File

    # Find a run on the Maxwell cluster
    run = open_run(proposal=700000, run=1)

    # Open a run with a directory path
    run = RunDirectory("/gpfs/exfel/exp/XMPL/201750/p700000/raw/r0001")

    # Open an individual file
    file = H5File("RAW-R0017-DA01-S00000.h5")

After this step, you'll use the same methods to get data whether you opened a
run or a file.

Load data into memory - see :ref:`data-by-source-and-key` for more::

    # Get a labelled array
    arr = run["SA3_XTD10_PES/ADC/1:network", "digitizers.channel_4_A.raw.samples"].xarray()

    # Get a pandas dataframe of 1D fields
    df = run.get_dataframe(fields=[
        ("*_XGM/*", "*.i[xy]Pos"),
        ("*_XGM/*", "*.photonFlux")
    ])

Iterate through data for each pulse train - see :ref:`data-by-train` for more::

    for train_id, data in run.select("*/DET/*", "image.data").trains():
        mod0 = data["FXE_DET_LPD1M-1/DET/0CH0:xtdf"]["image.data"]

These are not the only ways to get data: :doc:`reading_files` describes
various other options.

Documentation contents
----------------------

.. toctree::
   :caption: Tutorials and Examples
   :maxdepth: 2

   xpd_examples
   inspection
   iterate_trains
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
   <https://in.xfel.eu/readthedocs/docs/data-analysis-user-documentation/en/latest/>`_

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

