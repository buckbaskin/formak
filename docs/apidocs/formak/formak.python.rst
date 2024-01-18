:py:mod:`formak.python`
=======================

.. py:module:: formak.python

.. autodoc2-docstring:: formak.python
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Config <formak.python.Config>`
     - .. autodoc2-docstring:: formak.python.Config
          :summary:
   * - :py:obj:`BasicBlock <formak.python.BasicBlock>`
     - .. autodoc2-docstring:: formak.python.BasicBlock
          :summary:
   * - :py:obj:`Model <formak.python.Model>`
     - .. autodoc2-docstring:: formak.python.Model
          :summary:
   * - :py:obj:`SensorModel <formak.python.SensorModel>`
     - .. autodoc2-docstring:: formak.python.SensorModel
          :summary:
   * - :py:obj:`ExtendedKalmanFilter <formak.python.ExtendedKalmanFilter>`
     - .. autodoc2-docstring:: formak.python.ExtendedKalmanFilter
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`compile <formak.python.compile>`
     - .. autodoc2-docstring:: formak.python.compile
          :summary:
   * - :py:obj:`compile_ekf <formak.python.compile_ekf>`
     - .. autodoc2-docstring:: formak.python.compile_ekf
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`DEFAULT_MODULES <formak.python.DEFAULT_MODULES>`
     - .. autodoc2-docstring:: formak.python.DEFAULT_MODULES
          :summary:
   * - :py:obj:`StateAndCovariance <formak.python.StateAndCovariance>`
     - .. autodoc2-docstring:: formak.python.StateAndCovariance
          :summary:

API
~~~

.. py:data:: DEFAULT_MODULES
   :canonical: formak.python.DEFAULT_MODULES
   :value: ('scipy', 'numpy', 'math')

   .. autodoc2-docstring:: formak.python.DEFAULT_MODULES

.. py:class:: Config
   :canonical: formak.python.Config

   .. autodoc2-docstring:: formak.python.Config

   .. py:attribute:: common_subexpression_elimination
      :canonical: formak.python.Config.common_subexpression_elimination
      :type: bool
      :value: True

      .. autodoc2-docstring:: formak.python.Config.common_subexpression_elimination

   .. py:attribute:: python_modules
      :canonical: formak.python.Config.python_modules
      :value: None

      .. autodoc2-docstring:: formak.python.Config.python_modules

   .. py:attribute:: extra_validation
      :canonical: formak.python.Config.extra_validation
      :type: bool
      :value: False

      .. autodoc2-docstring:: formak.python.Config.extra_validation

   .. py:attribute:: max_dt_sec
      :canonical: formak.python.Config.max_dt_sec
      :type: float
      :value: 0.1

      .. autodoc2-docstring:: formak.python.Config.max_dt_sec

   .. py:attribute:: innovation_filtering
      :canonical: formak.python.Config.innovation_filtering
      :type: float
      :value: 5.0

      .. autodoc2-docstring:: formak.python.Config.innovation_filtering

.. py:class:: BasicBlock(*, arglist: typing.List[str], statements: typing.List[typing.Any], config: formak.python.Config)
   :canonical: formak.python.BasicBlock

   .. autodoc2-docstring:: formak.python.BasicBlock

   .. rubric:: Initialization

   .. autodoc2-docstring:: formak.python.BasicBlock.__init__

   .. py:method:: __len__()
      :canonical: formak.python.BasicBlock.__len__

      .. autodoc2-docstring:: formak.python.BasicBlock.__len__

   .. py:method:: _compile()
      :canonical: formak.python.BasicBlock._compile

      .. autodoc2-docstring:: formak.python.BasicBlock._compile

   .. py:method:: execute(*args, **kwargs)
      :canonical: formak.python.BasicBlock.execute

      .. autodoc2-docstring:: formak.python.BasicBlock.execute

.. py:class:: Model(symbolic_model, config, calibration_map=None)
   :canonical: formak.python.Model

   .. autodoc2-docstring:: formak.python.Model

   .. rubric:: Initialization

   .. autodoc2-docstring:: formak.python.Model.__init__

   .. py:method:: model(dt, state, control=None)
      :canonical: formak.python.Model.model

      .. autodoc2-docstring:: formak.python.Model.model

.. py:class:: SensorModel(state_model, sensor_model, calibration_map, config)
   :canonical: formak.python.SensorModel

   .. autodoc2-docstring:: formak.python.SensorModel

   .. rubric:: Initialization

   .. autodoc2-docstring:: formak.python.SensorModel.__init__

   .. py:method:: __len__()
      :canonical: formak.python.SensorModel.__len__

      .. autodoc2-docstring:: formak.python.SensorModel.__len__

   .. py:method:: model(state_vector)
      :canonical: formak.python.SensorModel.model

      .. autodoc2-docstring:: formak.python.SensorModel.model

.. py:data:: StateAndCovariance
   :canonical: formak.python.StateAndCovariance
   :value: 'namedtuple(...)'

   .. autodoc2-docstring:: formak.python.StateAndCovariance

.. py:class:: ExtendedKalmanFilter(state_model, process_noise: typing.Dict[typing.Union[sympy.Symbol, typing.Tuple[sympy.Symbol, sympy.Symbol]], float], sensor_models, sensor_noises: typing.Dict[typing.Union[sympy.Symbol, typing.Tuple[sympy.Symbol, sympy.Symbol]], float], config, calibration_map=None)
   :canonical: formak.python.ExtendedKalmanFilter

   .. autodoc2-docstring:: formak.python.ExtendedKalmanFilter

   .. rubric:: Initialization

   .. autodoc2-docstring:: formak.python.ExtendedKalmanFilter.__init__

   .. py:method:: _construct_process(state_model, process_noise, calibration_map, config)
      :canonical: formak.python.ExtendedKalmanFilter._construct_process

      .. autodoc2-docstring:: formak.python.ExtendedKalmanFilter._construct_process

   .. py:method:: _construct_sensors(state_model, sensor_models, sensor_noises, calibration_map, config)
      :canonical: formak.python.ExtendedKalmanFilter._construct_sensors

      .. autodoc2-docstring:: formak.python.ExtendedKalmanFilter._construct_sensors

   .. py:method:: make_reading(key, *, data=None, **kwargs)
      :canonical: formak.python.ExtendedKalmanFilter.make_reading

      .. autodoc2-docstring:: formak.python.ExtendedKalmanFilter.make_reading

   .. py:method:: process_jacobian(dt, state, control)
      :canonical: formak.python.ExtendedKalmanFilter.process_jacobian

      .. autodoc2-docstring:: formak.python.ExtendedKalmanFilter.process_jacobian

   .. py:method:: control_jacobian(dt, state, control)
      :canonical: formak.python.ExtendedKalmanFilter.control_jacobian

      .. autodoc2-docstring:: formak.python.ExtendedKalmanFilter.control_jacobian

   .. py:method:: sensor_jacobian(sensor_key, state)
      :canonical: formak.python.ExtendedKalmanFilter.sensor_jacobian

      .. autodoc2-docstring:: formak.python.ExtendedKalmanFilter.sensor_jacobian

   .. py:method:: process_model(dt, state, covariance, control=None)
      :canonical: formak.python.ExtendedKalmanFilter.process_model

      .. autodoc2-docstring:: formak.python.ExtendedKalmanFilter.process_model

   .. py:method:: remove_innovation(innovation, S_inv)
      :canonical: formak.python.ExtendedKalmanFilter.remove_innovation

      .. autodoc2-docstring:: formak.python.ExtendedKalmanFilter.remove_innovation

   .. py:method:: sensor_model(state, covariance, *, sensor_key, sensor_reading)
      :canonical: formak.python.ExtendedKalmanFilter.sensor_model

      .. autodoc2-docstring:: formak.python.ExtendedKalmanFilter.sensor_model

   .. py:method:: _flatten_scoring_params(params)
      :canonical: formak.python.ExtendedKalmanFilter._flatten_scoring_params

      .. autodoc2-docstring:: formak.python.ExtendedKalmanFilter._flatten_scoring_params

   .. py:method:: _inverse_flatten_scoring_params(flattened)
      :canonical: formak.python.ExtendedKalmanFilter._inverse_flatten_scoring_params

      .. autodoc2-docstring:: formak.python.ExtendedKalmanFilter._inverse_flatten_scoring_params

   .. py:method:: fit(X, y=None, sample_weight=None)
      :canonical: formak.python.ExtendedKalmanFilter.fit

      .. autodoc2-docstring:: formak.python.ExtendedKalmanFilter.fit

   .. py:method:: mahalanobis(X)
      :canonical: formak.python.ExtendedKalmanFilter.mahalanobis

      .. autodoc2-docstring:: formak.python.ExtendedKalmanFilter.mahalanobis

   .. py:method:: score(X, y=None, sample_weight=None, explain_score=False)
      :canonical: formak.python.ExtendedKalmanFilter.score

      .. autodoc2-docstring:: formak.python.ExtendedKalmanFilter.score

   .. py:method:: transform(X, include_states=False)
      :canonical: formak.python.ExtendedKalmanFilter.transform

      .. autodoc2-docstring:: formak.python.ExtendedKalmanFilter.transform

   .. py:method:: fit_transform(X, y=None)
      :canonical: formak.python.ExtendedKalmanFilter.fit_transform

      .. autodoc2-docstring:: formak.python.ExtendedKalmanFilter.fit_transform

   .. py:method:: get_params(deep=True) -> dict
      :canonical: formak.python.ExtendedKalmanFilter.get_params

      .. autodoc2-docstring:: formak.python.ExtendedKalmanFilter.get_params

   .. py:method:: set_params(**params)
      :canonical: formak.python.ExtendedKalmanFilter.set_params

      .. autodoc2-docstring:: formak.python.ExtendedKalmanFilter.set_params

.. py:function:: compile(symbolic_model, calibration_map=None, *, config=None)
   :canonical: formak.python.compile

   .. autodoc2-docstring:: formak.python.compile

.. py:function:: compile_ekf(state_model: formak.common.UiModelBase, process_noise: typing.Dict[typing.Union[sympy.Symbol, typing.Tuple[sympy.Symbol, sympy.Symbol]], float], sensor_models, sensor_noises, calibration_map=None, *, config=None)
   :canonical: formak.python.compile_ekf

   .. autodoc2-docstring:: formak.python.compile_ekf
