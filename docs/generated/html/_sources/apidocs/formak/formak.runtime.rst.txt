:py:mod:`formak.runtime`
========================

.. py:module:: formak.runtime

.. autodoc2-docstring:: formak.runtime
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`StampedReading <formak.runtime.StampedReading>`
     - .. autodoc2-docstring:: formak.runtime.StampedReading
          :summary:
   * - :py:obj:`ManagedFilter <formak.runtime.ManagedFilter>`
     - .. autodoc2-docstring:: formak.runtime.ManagedFilter
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`StateAndVariance <formak.runtime.StateAndVariance>`
     - .. autodoc2-docstring:: formak.runtime.StateAndVariance
          :summary:

API
~~~

.. py:class:: StampedReading(timestamp, sensor_key, *, _data=None, **kwargs)
   :canonical: formak.runtime.StampedReading

   .. autodoc2-docstring:: formak.runtime.StampedReading

   .. rubric:: Initialization

   .. autodoc2-docstring:: formak.runtime.StampedReading.__init__

   .. py:method:: from_data(timestamp, sensor_key, data)
      :canonical: formak.runtime.StampedReading.from_data
      :classmethod:

      .. autodoc2-docstring:: formak.runtime.StampedReading.from_data

.. py:data:: StateAndVariance
   :canonical: formak.runtime.StateAndVariance
   :value: 'namedtuple(...)'

   .. autodoc2-docstring:: formak.runtime.StateAndVariance

.. py:class:: ManagedFilter(ekf, start_time: float, state, covariance, calibration_map=None)
   :canonical: formak.runtime.ManagedFilter

   .. autodoc2-docstring:: formak.runtime.ManagedFilter

   .. rubric:: Initialization

   .. autodoc2-docstring:: formak.runtime.ManagedFilter.__init__

   .. py:method:: tick(output_time: float, *, control=None, readings: typing.Optional[typing.List[formak.runtime.StampedReading]] = None)
      :canonical: formak.runtime.ManagedFilter.tick

      .. autodoc2-docstring:: formak.runtime.ManagedFilter.tick

   .. py:method:: _process_model(output_time, control)
      :canonical: formak.runtime.ManagedFilter._process_model

      .. autodoc2-docstring:: formak.runtime.ManagedFilter._process_model
