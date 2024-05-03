Multi-module detector data
==========================

.. module:: extra_data.components

Several X-ray pixel detectors are composed of multiple modules, which are
stored as separate sources at EuXFEL.
``extra_data`` includes convenient interfaces to access data from AGIPD, LPD,
DSSC and JUNGFRAU, pulling together the separate modules into a single array.

.. note::

   These detectors can record a lot of data, so loading it all into memory at
   once may be impossible. You can use the :meth:`~LPD1M.split_trains` method
   to work on a chunk at a time, or work with Dask arrays.

.. autoclass:: AGIPD1M

   The methods of this class are identical to those of :class:`LPD1M`, below.

.. autoclass:: AGIPD500K

.. autoclass:: DSSC1M

   The methods of this class are identical to those of :class:`LPD1M`, below.

.. autoclass:: LPD1M

   Selecting a key from the detector, e.g. ``det['image.data']``, gives an
   object similar to a single-source :class:`KeyData`, but with the modules
   arranged along the first axis. So ``det['image.data'].ndarray()`` will
   load all the selected data as a NumPy array.

   .. automethod:: masked_data

   .. automethod:: get_array

   .. automethod:: get_dask_array

   .. automethod:: trains

   .. automethod:: data_availability

   .. automethod:: select_trains

   .. automethod:: split_trains

   .. automethod:: write_frames

   .. automethod:: write_virtual_cxi

.. seealso::

   :doc:`lpd_data`: An example using the class above.

.. autoclass:: JUNGFRAU

   Selecting a key from the detector, e.g. ``jf['data.adc']``, gives an
   object similar to a single-source :class:`KeyData`, but with the modules
   arranged along the first axis. So ``jf['data.adc'].ndarray()`` will
   load all the selected data as a NumPy array.

   .. automethod:: masked_data

   .. automethod:: get_array

   .. automethod:: get_dask_array

   .. automethod:: trains

   .. automethod:: data_availability

   .. automethod:: select_trains

   .. automethod:: split_trains

   .. automethod:: write_virtual_cxi

.. autofunction:: identify_multimod_detectors

If you get data for a train from the main :class:`DataCollection` interface,
there is also another way to combine detector modules from AGIPD, DSSC or LPD:

.. currentmodule:: extra_data

.. autofunction:: stack_detector_data
