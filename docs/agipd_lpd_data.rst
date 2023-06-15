Multi-module detector data
==========================

.. module:: extra_data.components

Several X-ray pixel detectors are composed of multiple modules, which are
stored as separate sources at EuXFEL.
``extra_data`` includes convenient interfaces to access data from AGIPD, LPD,
DSSC and JUNGFRAU, pulling together the separate modules into a single array.

.. note::

   These detectors can record a lot of data. The ``.get_array()`` method
   loads all of the selected data into memory, which may not be practical for
   entire runs. You might need to think about iterating over trains, selecting
   batches of trains from the run, or using `Dask arrays
   <https://docs.dask.org/en/latest/array.html>`_.


.. autoclass:: AGIPD1M

   The methods of this class are identical to those of :class:`LPD1M`, below.

.. autoclass:: AGIPD500K

.. autoclass:: DSSC1M

   The methods of this class are identical to those of :class:`LPD1M`, below.

.. autoclass:: LPD1M

   .. automethod:: get_array

   .. automethod:: get_dask_array

   .. automethod:: trains

   .. automethod:: select_trains

   .. automethod:: split_trains

   .. automethod:: write_frames

   .. automethod:: write_virtual_cxi

.. seealso::

   :doc:`lpd_data`: An example using the class above.

.. autoclass:: JUNGFRAU

   .. automethod:: get_array

   .. automethod:: get_dask_array

   .. automethod:: trains

   .. automethod:: select_trains

   .. automethod:: split_trains

   .. automethod:: write_virtual_cxi

.. autofunction:: identify_multimod_detectors

If you get data for a train from the main :class:`DataCollection` interface,
there is also another way to combine detector modules from AGIPD, DSSC or LPD:

.. currentmodule:: extra_data

.. autofunction:: stack_detector_data
