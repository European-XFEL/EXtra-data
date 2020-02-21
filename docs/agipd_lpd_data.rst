AGIPD, LPD & DSSC data
======================

.. module:: extra_data.components

These data from AGIPD, LPD and DSSC is spread out in separate files.
``extra_data`` includes convenient interfaces to access this data,
pulling together the separate modules into a single array.

.. autoclass:: AGIPD1M

   The methods of this class are identical to those of :class:`LPD1M`, below.

.. autoclass:: DSSC1M

   The methods of this class are identical to those of :class:`LPD1M`, below.

.. autoclass:: LPD1M

   .. automethod:: get_dask_array

   .. automethod:: get_array

   .. automethod:: trains

   .. automethod:: write_virtual_cxi

.. seealso::

   :doc:`lpd_data`: An example using the class above.

If you get data for a train from the main :class:`DataCollection` interface,
there is also another way to combine detector modules from AGIPD or LPD:

.. currentmodule:: extra_data

.. autofunction:: stack_detector_data
