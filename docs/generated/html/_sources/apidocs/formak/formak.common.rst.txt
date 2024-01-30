:py:mod:`formak.common`
=======================

.. py:module:: formak.common

.. autodoc2-docstring:: formak.common
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`UiModelBase <formak.common.UiModelBase>`
     - .. autodoc2-docstring:: formak.common.UiModelBase
          :summary:
   * - :py:obj:`_NamedArrayBase <formak.common._NamedArrayBase>`
     -

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`model_validation <formak.common.model_validation>`
     - .. autodoc2-docstring:: formak.common.model_validation
          :summary:
   * - :py:obj:`named_vector <formak.common.named_vector>`
     - .. autodoc2-docstring:: formak.common.named_vector
          :summary:
   * - :py:obj:`named_covariance <formak.common.named_covariance>`
     - .. autodoc2-docstring:: formak.common.named_covariance
          :summary:
   * - :py:obj:`plot_pair <formak.common.plot_pair>`
     - .. autodoc2-docstring:: formak.common.plot_pair
          :summary:
   * - :py:obj:`plot_quaternion_timeseries <formak.common.plot_quaternion_timeseries>`
     - .. autodoc2-docstring:: formak.common.plot_quaternion_timeseries
          :summary:

API
~~~

.. py:class:: UiModelBase
   :canonical: formak.common.UiModelBase

   .. autodoc2-docstring:: formak.common.UiModelBase

.. py:function:: model_validation(state_model, process_noise: typing.Dict[typing.Union[sympy.Symbol, typing.Tuple[sympy.Symbol, sympy.Symbol]], float], sensor_models, *, verbose=True, extra_validation=False, calibration_map: typing.Dict[sympy.Symbol, float])
   :canonical: formak.common.model_validation

   .. autodoc2-docstring:: formak.common.model_validation

.. py:class:: _NamedArrayBase(name: str, kwargs: typing.Dict[typing.Any, typing.Any])
   :canonical: formak.common._NamedArrayBase

   Bases: :py:obj:`abc.ABC`

   .. py:method:: __repr__()
      :canonical: formak.common._NamedArrayBase.__repr__

   .. py:method:: __iter__()
      :canonical: formak.common._NamedArrayBase.__iter__

      .. autodoc2-docstring:: formak.common._NamedArrayBase.__iter__

   .. py:method:: from_data(data)
      :canonical: formak.common._NamedArrayBase.from_data
      :classmethod:

      .. autodoc2-docstring:: formak.common._NamedArrayBase.from_data

   .. py:method:: from_dict(mapping)
      :canonical: formak.common._NamedArrayBase.from_dict
      :classmethod:

      .. autodoc2-docstring:: formak.common._NamedArrayBase.from_dict

   .. py:method:: __subclasshook__(Other)
      :canonical: formak.common._NamedArrayBase.__subclasshook__
      :abstractmethod:
      :classmethod:

.. py:function:: named_vector(name, arglist)
   :canonical: formak.common.named_vector

   .. autodoc2-docstring:: formak.common.named_vector

.. py:function:: named_covariance(name, arglist)
   :canonical: formak.common.named_covariance

   .. autodoc2-docstring:: formak.common.named_covariance

.. py:function:: plot_pair(*, states, expected_states, arglist, x_name, y_name, file_id)
   :canonical: formak.common.plot_pair

   .. autodoc2-docstring:: formak.common.plot_pair

.. py:function:: plot_quaternion_timeseries(*, times, states, expected_states, arglist, x_name, file_id)
   :canonical: formak.common.plot_quaternion_timeseries

   .. autodoc2-docstring:: formak.common.plot_quaternion_timeseries
