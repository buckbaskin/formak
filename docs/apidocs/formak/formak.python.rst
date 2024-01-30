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
   * - :py:obj:`SklearnEKFAdapter <formak.python.SklearnEKFAdapter>`
     - .. autodoc2-docstring:: formak.python.SklearnEKFAdapter
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`assert_valid_covariance <formak.python.assert_valid_covariance>`
     - .. autodoc2-docstring:: formak.python.assert_valid_covariance
          :summary:
   * - :py:obj:`nearest_positive_definite <formak.python.nearest_positive_definite>`
     - .. autodoc2-docstring:: formak.python.nearest_positive_definite
          :summary:
   * - :py:obj:`compile <formak.python.compile>`
     - .. autodoc2-docstring:: formak.python.compile
          :summary:
   * - :py:obj:`compile_ekf <formak.python.compile_ekf>`
     - .. autodoc2-docstring:: formak.python.compile_ekf
          :summary:
   * - :py:obj:`force_to_ndarray <formak.python.force_to_ndarray>`
     - .. autodoc2-docstring:: formak.python.force_to_ndarray
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
      :type: tuple[typing.Any, typing.Any, typing.Any, typing.Any]
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
      :type: float | None
      :value: 5.0

      .. autodoc2-docstring:: formak.python.Config.innovation_filtering

.. py:class:: BasicBlock(*, arglist: list[str], statements: list[typing.Any], config: formak.python.Config)
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

.. py:function:: assert_valid_covariance(covariance: numpy.typing.NDArray, *, name: str = 'Covariance', negative_tol: float = -1e-15)
   :canonical: formak.python.assert_valid_covariance

   .. autodoc2-docstring:: formak.python.assert_valid_covariance

.. py:function:: nearest_positive_definite(covariance: dict[sympy.Symbol | tuple[sympy.Symbol, sympy.Symbol], float])
   :canonical: formak.python.nearest_positive_definite

   .. autodoc2-docstring:: formak.python.nearest_positive_definite

.. py:class:: ExtendedKalmanFilter(state_model, process_noise: dict[sympy.Symbol | tuple[sympy.Symbol, sympy.Symbol], float], sensor_models: dict[str, sympy.core.expr.Expr], sensor_noises: dict[str, dict[sympy.Symbol | tuple[sympy.Symbol, sympy.Symbol], float]], config: formak.python.Config, calibration_map: dict[sympy.Symbol, float] | None = None)
   :canonical: formak.python.ExtendedKalmanFilter

   .. autodoc2-docstring:: formak.python.ExtendedKalmanFilter

   .. rubric:: Initialization

   .. autodoc2-docstring:: formak.python.ExtendedKalmanFilter.__init__

   .. py:method:: _construct_process(state_model, process_noise: dict[sympy.Symbol | tuple[sympy.Symbol, sympy.Symbol], float], calibration_map: dict[sympy.Symbol, float], config: formak.python.Config) -> None
      :canonical: formak.python.ExtendedKalmanFilter._construct_process

      .. autodoc2-docstring:: formak.python.ExtendedKalmanFilter._construct_process

   .. py:method:: _construct_sensors(state_model: formak.common.UiModelBase, sensor_models: dict[str, sympy.core.expr.Expr], sensor_noises: dict[str, dict[sympy.Symbol | tuple[sympy.Symbol, sympy.Symbol], float]], calibration_map: dict[sympy.Symbol, float], config: formak.python.Config) -> None
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

   .. py:method:: remove_innovation(innovation: numpy.typing.NDArray, S_inv: numpy.typing.NDArray) -> bool
      :canonical: formak.python.ExtendedKalmanFilter.remove_innovation

      .. autodoc2-docstring:: formak.python.ExtendedKalmanFilter.remove_innovation

   .. py:method:: sensor_model(state, covariance, *, sensor_key, sensor_reading)
      :canonical: formak.python.ExtendedKalmanFilter.sensor_model

      .. autodoc2-docstring:: formak.python.ExtendedKalmanFilter.sensor_model

.. py:function:: compile(symbolic_model, calibration_map=None, *, config=None)
   :canonical: formak.python.compile

   .. autodoc2-docstring:: formak.python.compile

.. py:function:: compile_ekf(symbolic_model: formak.common.UiModelBase, process_noise: dict[sympy.Symbol | tuple[sympy.Symbol, sympy.Symbol], float], sensor_models: dict[str, sympy.core.expr.Expr], sensor_noises, calibration_map: dict[sympy.Symbol, float] | None = None, *, config=None) -> formak.python.ExtendedKalmanFilter
   :canonical: formak.python.compile_ekf

   .. autodoc2-docstring:: formak.python.compile_ekf

.. py:function:: force_to_ndarray(mat: typing.Any) -> numpy.typing.NDArray | None
   :canonical: formak.python.force_to_ndarray

   .. autodoc2-docstring:: formak.python.force_to_ndarray

.. py:class:: SklearnEKFAdapter(symbolic_model: formak.common.UiModelBase | None = None, process_noise: dict[sympy.Symbol | tuple[sympy.Symbol, sympy.Symbol], float] | None = None, sensor_models: dict[sympy.Symbol, sympy.core.expr.Expr] | None = None, sensor_noises: dict[sympy.Symbol | tuple[sympy.Symbol, sympy.Symbol], float] | None = None, calibration_map: dict[sympy.Symbol, float] | None = None, *, config: formak.python.Config | None = None)
   :canonical: formak.python.SklearnEKFAdapter

   Bases: :py:obj:`sklearn.base.BaseEstimator`

   .. autodoc2-docstring:: formak.python.SklearnEKFAdapter

   .. rubric:: Initialization

   .. autodoc2-docstring:: formak.python.SklearnEKFAdapter.__init__

   .. py:attribute:: allowed_keys
      :canonical: formak.python.SklearnEKFAdapter.allowed_keys
      :value: ['symbolic_model', 'process_noise', 'sensor_models', 'sensor_noises', 'calibration_map', 'config']

      .. autodoc2-docstring:: formak.python.SklearnEKFAdapter.allowed_keys

   .. py:method:: Create(symbolic_model: formak.common.UiModelBase, process_noise: dict[sympy.Symbol | tuple[sympy.Symbol, sympy.Symbol], float], sensor_models: dict[str, sympy.core.expr.Expr], sensor_noises: dict[str, dict[sympy.Symbol | tuple[sympy.Symbol, sympy.Symbol], float]], calibration_map: dict[sympy.Symbol, float] | None = None, *, config: formak.python.Config | None = None)
      :canonical: formak.python.SklearnEKFAdapter.Create
      :classmethod:

      .. autodoc2-docstring:: formak.python.SklearnEKFAdapter.Create

   .. py:method:: _flatten_process_noise(process_noise: dict[sympy.Symbol | tuple[sympy.Symbol, sympy.Symbol], float])
      :canonical: formak.python.SklearnEKFAdapter._flatten_process_noise

      .. autodoc2-docstring:: formak.python.SklearnEKFAdapter._flatten_process_noise

   .. py:method:: _sensor_noise_to_array(sensor_noises: dict[str, dict[sympy.Symbol | tuple[sympy.Symbol, sympy.Symbol], float]])
      :canonical: formak.python.SklearnEKFAdapter._sensor_noise_to_array

      .. autodoc2-docstring:: formak.python.SklearnEKFAdapter._sensor_noise_to_array

   .. py:method:: _compile_sensor_models(sensor_models: dict[str, sympy.core.expr.Expr])
      :canonical: formak.python.SklearnEKFAdapter._compile_sensor_models

      .. autodoc2-docstring:: formak.python.SklearnEKFAdapter._compile_sensor_models

   .. py:method:: _flatten_dict_diagonal(mapping: dict[sympy.Symbol | tuple[sympy.Symbol, sympy.Symbol], float], arglist: list[sympy.Symbol]) -> typing.Iterator[float]
      :canonical: formak.python.SklearnEKFAdapter._flatten_dict_diagonal

      .. autodoc2-docstring:: formak.python.SklearnEKFAdapter._flatten_dict_diagonal

   .. py:method:: _inverse_flatten_dict_diagonal(vector, arglist) -> dict[sympy.Symbol | tuple[sympy.Symbol, sympy.Symbol], float]
      :canonical: formak.python.SklearnEKFAdapter._inverse_flatten_dict_diagonal

      .. autodoc2-docstring:: formak.python.SklearnEKFAdapter._inverse_flatten_dict_diagonal

   .. py:method:: _flatten_scoring_params() -> list[float]
      :canonical: formak.python.SklearnEKFAdapter._flatten_scoring_params

      .. autodoc2-docstring:: formak.python.SklearnEKFAdapter._flatten_scoring_params

   .. py:method:: _inverse_flatten_scoring_params(flattened: list[float]) -> dict[str, typing.Any]
      :canonical: formak.python.SklearnEKFAdapter._inverse_flatten_scoring_params

      .. autodoc2-docstring:: formak.python.SklearnEKFAdapter._inverse_flatten_scoring_params

   .. py:method:: fit(X: typing.Any, y: typing.Any | None = None, sample_weight: numpy.typing.NDArray | None = None) -> formak.python.SklearnEKFAdapter
      :canonical: formak.python.SklearnEKFAdapter.fit

      .. autodoc2-docstring:: formak.python.SklearnEKFAdapter.fit

   .. py:method:: mahalanobis(X: typing.Any) -> numpy.typing.NDArray
      :canonical: formak.python.SklearnEKFAdapter.mahalanobis

      .. autodoc2-docstring:: formak.python.SklearnEKFAdapter.mahalanobis

   .. py:method:: score(X: typing.Any, y: typing.Any | None = None, sample_weight: typing.Any | None = None, explain_score: bool = False) -> float | tuple[float, tuple[float, float, float, float, float, float]]
      :canonical: formak.python.SklearnEKFAdapter.score

      .. autodoc2-docstring:: formak.python.SklearnEKFAdapter.score

   .. py:method:: transform(X: typing.Any, include_states=False) -> numpy.typing.NDArray | tuple[numpy.typing.NDArray, numpy.typing.NDArray, numpy.typing.NDArray]
      :canonical: formak.python.SklearnEKFAdapter.transform

      .. autodoc2-docstring:: formak.python.SklearnEKFAdapter.transform

   .. py:method:: fit_transform(X, y=None) -> numpy.typing.NDArray | tuple[numpy.typing.NDArray, numpy.typing.NDArray, numpy.typing.NDArray]
      :canonical: formak.python.SklearnEKFAdapter.fit_transform

      .. autodoc2-docstring:: formak.python.SklearnEKFAdapter.fit_transform

   .. py:method:: get_params(deep: bool = True) -> dict[str, typing.Any]
      :canonical: formak.python.SklearnEKFAdapter.get_params

      .. autodoc2-docstring:: formak.python.SklearnEKFAdapter.get_params

   .. py:method:: set_params(**params) -> formak.python.SklearnEKFAdapter
      :canonical: formak.python.SklearnEKFAdapter.set_params

      .. autodoc2-docstring:: formak.python.SklearnEKFAdapter.set_params

   .. py:method:: export_python() -> formak.python.ExtendedKalmanFilter
      :canonical: formak.python.SklearnEKFAdapter.export_python

      .. autodoc2-docstring:: formak.python.SklearnEKFAdapter.export_python
