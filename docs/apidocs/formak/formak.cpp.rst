:py:mod:`formak.cpp`
====================

.. py:module:: formak.cpp

.. autodoc2-docstring:: formak.cpp
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Config <formak.cpp.Config>`
     - .. autodoc2-docstring:: formak.cpp.Config
          :summary:
   * - :py:obj:`CppCompileResult <formak.cpp.CppCompileResult>`
     - .. autodoc2-docstring:: formak.cpp.CppCompileResult
          :summary:
   * - :py:obj:`BasicBlock <formak.cpp.BasicBlock>`
     - .. autodoc2-docstring:: formak.cpp.BasicBlock
          :summary:
   * - :py:obj:`Model <formak.cpp.Model>`
     - .. autodoc2-docstring:: formak.cpp.Model
          :summary:
   * - :py:obj:`ExtendedKalmanFilter <formak.cpp.ExtendedKalmanFilter>`
     - .. autodoc2-docstring:: formak.cpp.ExtendedKalmanFilter
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`_generate_model_function_bodies <formak.cpp._generate_model_function_bodies>`
     - .. autodoc2-docstring:: formak.cpp._generate_model_function_bodies
          :summary:
   * - :py:obj:`_generate_ekf_function_bodies <formak.cpp._generate_ekf_function_bodies>`
     - .. autodoc2-docstring:: formak.cpp._generate_ekf_function_bodies
          :summary:
   * - :py:obj:`_compile_argparse <formak.cpp._compile_argparse>`
     - .. autodoc2-docstring:: formak.cpp._compile_argparse
          :summary:
   * - :py:obj:`_header_body <formak.cpp._header_body>`
     - .. autodoc2-docstring:: formak.cpp._header_body
          :summary:
   * - :py:obj:`header_from_ast <formak.cpp.header_from_ast>`
     - .. autodoc2-docstring:: formak.cpp.header_from_ast
          :summary:
   * - :py:obj:`_source_body <formak.cpp._source_body>`
     - .. autodoc2-docstring:: formak.cpp._source_body
          :summary:
   * - :py:obj:`source_from_ast <formak.cpp.source_from_ast>`
     - .. autodoc2-docstring:: formak.cpp.source_from_ast
          :summary:
   * - :py:obj:`_compile_impl <formak.cpp._compile_impl>`
     - .. autodoc2-docstring:: formak.cpp._compile_impl
          :summary:
   * - :py:obj:`compile <formak.cpp.compile>`
     - .. autodoc2-docstring:: formak.cpp.compile
          :summary:
   * - :py:obj:`compile_ekf <formak.cpp.compile_ekf>`
     - .. autodoc2-docstring:: formak.cpp.compile_ekf
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`DEFAULT_MODULES <formak.cpp.DEFAULT_MODULES>`
     - .. autodoc2-docstring:: formak.cpp.DEFAULT_MODULES
          :summary:
   * - :py:obj:`logger <formak.cpp.logger>`
     - .. autodoc2-docstring:: formak.cpp.logger
          :summary:
   * - :py:obj:`ReadingT <formak.cpp.ReadingT>`
     - .. autodoc2-docstring:: formak.cpp.ReadingT
          :summary:

API
~~~

.. py:data:: DEFAULT_MODULES
   :canonical: formak.cpp.DEFAULT_MODULES
   :value: ('scipy', 'numpy', 'math')

   .. autodoc2-docstring:: formak.cpp.DEFAULT_MODULES

.. py:data:: logger
   :canonical: formak.cpp.logger
   :value: 'getLogger(...)'

   .. autodoc2-docstring:: formak.cpp.logger

.. py:class:: Config
   :canonical: formak.cpp.Config

   .. autodoc2-docstring:: formak.cpp.Config

   .. py:attribute:: common_subexpression_elimination
      :canonical: formak.cpp.Config.common_subexpression_elimination
      :type: bool
      :value: True

      .. autodoc2-docstring:: formak.cpp.Config.common_subexpression_elimination

   .. py:attribute:: extra_validation
      :canonical: formak.cpp.Config.extra_validation
      :type: bool
      :value: False

      .. autodoc2-docstring:: formak.cpp.Config.extra_validation

   .. py:attribute:: max_dt_sec
      :canonical: formak.cpp.Config.max_dt_sec
      :type: float
      :value: 0.1

      .. autodoc2-docstring:: formak.cpp.Config.max_dt_sec

   .. py:attribute:: innovation_filtering
      :canonical: formak.cpp.Config.innovation_filtering
      :type: float
      :value: 5.0

      .. autodoc2-docstring:: formak.cpp.Config.innovation_filtering

   .. py:method:: ccode()
      :canonical: formak.cpp.Config.ccode

      .. autodoc2-docstring:: formak.cpp.Config.ccode

.. py:class:: CppCompileResult
   :canonical: formak.cpp.CppCompileResult

   .. autodoc2-docstring:: formak.cpp.CppCompileResult

   .. py:attribute:: success
      :canonical: formak.cpp.CppCompileResult.success
      :type: bool
      :value: None

      .. autodoc2-docstring:: formak.cpp.CppCompileResult.success

   .. py:attribute:: header_path
      :canonical: formak.cpp.CppCompileResult.header_path
      :type: typing.Optional[str]
      :value: None

      .. autodoc2-docstring:: formak.cpp.CppCompileResult.header_path

   .. py:attribute:: source_path
      :canonical: formak.cpp.CppCompileResult.source_path
      :type: typing.Optional[str]
      :value: None

      .. autodoc2-docstring:: formak.cpp.CppCompileResult.source_path

.. py:class:: BasicBlock(*, statements: typing.List[typing.Tuple[str, typing.Any]], indent: int, config: formak.cpp.Config)
   :canonical: formak.cpp.BasicBlock

   .. autodoc2-docstring:: formak.cpp.BasicBlock

   .. rubric:: Initialization

   .. autodoc2-docstring:: formak.cpp.BasicBlock.__init__

   .. py:method:: __len__()
      :canonical: formak.cpp.BasicBlock.__len__

      .. autodoc2-docstring:: formak.cpp.BasicBlock.__len__

   .. py:method:: compile()
      :canonical: formak.cpp.BasicBlock.compile

      .. autodoc2-docstring:: formak.cpp.BasicBlock.compile

.. py:class:: Model(symbolic_model, calibration_map, namespace, header_include, config)
   :canonical: formak.cpp.Model

   .. autodoc2-docstring:: formak.cpp.Model

   .. rubric:: Initialization

   .. autodoc2-docstring:: formak.cpp.Model.__init__

   .. py:method:: _translate_model(symbolic_model)
      :canonical: formak.cpp.Model._translate_model

      .. autodoc2-docstring:: formak.cpp.Model._translate_model

   .. py:method:: _translate_return()
      :canonical: formak.cpp.Model._translate_return

      .. autodoc2-docstring:: formak.cpp.Model._translate_return

   .. py:method:: model_body()
      :canonical: formak.cpp.Model.model_body

      .. autodoc2-docstring:: formak.cpp.Model.model_body

   .. py:method:: enable_control()
      :canonical: formak.cpp.Model.enable_control

      .. autodoc2-docstring:: formak.cpp.Model.enable_control

   .. py:method:: enable_calibration()
      :canonical: formak.cpp.Model.enable_calibration

      .. autodoc2-docstring:: formak.cpp.Model.enable_calibration

.. py:data:: ReadingT
   :canonical: formak.cpp.ReadingT
   :value: 'namedtuple(...)'

   .. autodoc2-docstring:: formak.cpp.ReadingT

.. py:class:: ExtendedKalmanFilter(state_model, process_noise, sensor_models, sensor_noises, namespace, header_include, config, calibration_map=None)
   :canonical: formak.cpp.ExtendedKalmanFilter

   .. autodoc2-docstring:: formak.cpp.ExtendedKalmanFilter

   .. rubric:: Initialization

   .. autodoc2-docstring:: formak.cpp.ExtendedKalmanFilter.__init__

   .. py:method:: _translate_process_model(symbolic_model)
      :canonical: formak.cpp.ExtendedKalmanFilter._translate_process_model

      .. autodoc2-docstring:: formak.cpp.ExtendedKalmanFilter._translate_process_model

   .. py:method:: process_model_body()
      :canonical: formak.cpp.ExtendedKalmanFilter.process_model_body

      .. autodoc2-docstring:: formak.cpp.ExtendedKalmanFilter.process_model_body

   .. py:method:: _translate_process_jacobian(symbolic_model)
      :canonical: formak.cpp.ExtendedKalmanFilter._translate_process_jacobian

      .. autodoc2-docstring:: formak.cpp.ExtendedKalmanFilter._translate_process_jacobian

   .. py:method:: process_jacobian_body()
      :canonical: formak.cpp.ExtendedKalmanFilter.process_jacobian_body

      .. autodoc2-docstring:: formak.cpp.ExtendedKalmanFilter.process_jacobian_body

   .. py:method:: _translate_control_jacobian(symbolic_model)
      :canonical: formak.cpp.ExtendedKalmanFilter._translate_control_jacobian

      .. autodoc2-docstring:: formak.cpp.ExtendedKalmanFilter._translate_control_jacobian

   .. py:method:: control_jacobian_body()
      :canonical: formak.cpp.ExtendedKalmanFilter.control_jacobian_body

      .. autodoc2-docstring:: formak.cpp.ExtendedKalmanFilter.control_jacobian_body

   .. py:method:: _translate_return()
      :canonical: formak.cpp.ExtendedKalmanFilter._translate_return

      .. autodoc2-docstring:: formak.cpp.ExtendedKalmanFilter._translate_return

   .. py:method:: enable_control()
      :canonical: formak.cpp.ExtendedKalmanFilter.enable_control

      .. autodoc2-docstring:: formak.cpp.ExtendedKalmanFilter.enable_control

   .. py:method:: _translate_control_covariance(covariance)
      :canonical: formak.cpp.ExtendedKalmanFilter._translate_control_covariance

      .. autodoc2-docstring:: formak.cpp.ExtendedKalmanFilter._translate_control_covariance

   .. py:method:: control_covariance_body()
      :canonical: formak.cpp.ExtendedKalmanFilter.control_covariance_body

      .. autodoc2-docstring:: formak.cpp.ExtendedKalmanFilter.control_covariance_body

   .. py:method:: enable_calibration()
      :canonical: formak.cpp.ExtendedKalmanFilter.enable_calibration

      .. autodoc2-docstring:: formak.cpp.ExtendedKalmanFilter.enable_calibration

   .. py:method:: _translate_sensor_model(sensor_model_mapping)
      :canonical: formak.cpp.ExtendedKalmanFilter._translate_sensor_model

      .. autodoc2-docstring:: formak.cpp.ExtendedKalmanFilter._translate_sensor_model

   .. py:method:: reading_types(verbose=False)
      :canonical: formak.cpp.ExtendedKalmanFilter.reading_types

      .. autodoc2-docstring:: formak.cpp.ExtendedKalmanFilter.reading_types

   .. py:method:: _translate_sensor_jacobian_impl(sensor_model_mapping)
      :canonical: formak.cpp.ExtendedKalmanFilter._translate_sensor_jacobian_impl

      .. autodoc2-docstring:: formak.cpp.ExtendedKalmanFilter._translate_sensor_jacobian_impl

   .. py:method:: _translate_sensor_jacobian(typename, sensor_model_mapping)
      :canonical: formak.cpp.ExtendedKalmanFilter._translate_sensor_jacobian

      .. autodoc2-docstring:: formak.cpp.ExtendedKalmanFilter._translate_sensor_jacobian

   .. py:method:: _translate_sensor_covariance_impl(covariance)
      :canonical: formak.cpp.ExtendedKalmanFilter._translate_sensor_covariance_impl

      .. autodoc2-docstring:: formak.cpp.ExtendedKalmanFilter._translate_sensor_covariance_impl

   .. py:method:: _translate_sensor_covariance(typename, covariance)
      :canonical: formak.cpp.ExtendedKalmanFilter._translate_sensor_covariance

      .. autodoc2-docstring:: formak.cpp.ExtendedKalmanFilter._translate_sensor_covariance

.. py:function:: _generate_model_function_bodies(header_location, namespace, symbolic_model, calibration_map, config)
   :canonical: formak.cpp._generate_model_function_bodies

   .. autodoc2-docstring:: formak.cpp._generate_model_function_bodies

.. py:function:: _generate_ekf_function_bodies(header_location, namespace, state_model, process_noise, sensor_models, sensor_noises, calibration_map, config)
   :canonical: formak.cpp._generate_ekf_function_bodies

   .. autodoc2-docstring:: formak.cpp._generate_ekf_function_bodies

.. py:function:: _compile_argparse()
   :canonical: formak.cpp._compile_argparse

   .. autodoc2-docstring:: formak.cpp._compile_argparse

.. py:function:: _header_body(*, generator) -> typing.Iterable[formak.ast_tools.BaseAst]
   :canonical: formak.cpp._header_body

   .. autodoc2-docstring:: formak.cpp._header_body

.. py:function:: header_from_ast(*, generator) -> str
   :canonical: formak.cpp.header_from_ast

   .. autodoc2-docstring:: formak.cpp.header_from_ast

.. py:function:: _source_body(*, generator)
   :canonical: formak.cpp._source_body

   .. autodoc2-docstring:: formak.cpp._source_body

.. py:function:: source_from_ast(*, generator)
   :canonical: formak.cpp.source_from_ast

   .. autodoc2-docstring:: formak.cpp.source_from_ast

.. py:function:: _compile_impl(args, *, generator)
   :canonical: formak.cpp._compile_impl

   .. autodoc2-docstring:: formak.cpp._compile_impl

.. py:function:: compile(symbolic_model, calibration_map=None, *, config=None)
   :canonical: formak.cpp.compile

   .. autodoc2-docstring:: formak.cpp.compile

.. py:function:: compile_ekf(state_model, process_noise, sensor_models, sensor_noises, calibration_map=None, *, config=None)
   :canonical: formak.cpp.compile_ekf

   .. autodoc2-docstring:: formak.cpp.compile_ekf
