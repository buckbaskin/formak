:py:mod:`formak.ui_state_machine`
=================================

.. py:module:: formak.ui_state_machine

.. autodoc2-docstring:: formak.ui_state_machine
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`StateId <formak.ui_state_machine.StateId>`
     -
   * - :py:obj:`ConfigView <formak.ui_state_machine.ConfigView>`
     - .. autodoc2-docstring:: formak.ui_state_machine.ConfigView
          :summary:
   * - :py:obj:`StateMachineState <formak.ui_state_machine.StateMachineState>`
     - .. autodoc2-docstring:: formak.ui_state_machine.StateMachineState
          :summary:
   * - :py:obj:`NisScore <formak.ui_state_machine.NisScore>`
     - .. autodoc2-docstring:: formak.ui_state_machine.NisScore
          :summary:
   * - :py:obj:`FitModelState <formak.ui_state_machine.FitModelState>`
     - .. autodoc2-docstring:: formak.ui_state_machine.FitModelState
          :summary:
   * - :py:obj:`SymbolicModelState <formak.ui_state_machine.SymbolicModelState>`
     - .. autodoc2-docstring:: formak.ui_state_machine.SymbolicModelState
          :summary:
   * - :py:obj:`DesignManager <formak.ui_state_machine.DesignManager>`
     - .. autodoc2-docstring:: formak.ui_state_machine.DesignManager
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`SearchState <formak.ui_state_machine.SearchState>`
     - .. autodoc2-docstring:: formak.ui_state_machine.SearchState
          :summary:
   * - :py:obj:`PIPELINE_STAGE_NAME <formak.ui_state_machine.PIPELINE_STAGE_NAME>`
     - .. autodoc2-docstring:: formak.ui_state_machine.PIPELINE_STAGE_NAME
          :summary:

API
~~~

.. py:data:: SearchState
   :canonical: formak.ui_state_machine.SearchState
   :value: 'namedtuple(...)'

   .. autodoc2-docstring:: formak.ui_state_machine.SearchState

.. py:class:: StateId
   :canonical: formak.ui_state_machine.StateId

   Bases: :py:obj:`enum.Enum`

   .. py:attribute:: Start
      :canonical: formak.ui_state_machine.StateId.Start
      :value: 0

      .. autodoc2-docstring:: formak.ui_state_machine.StateId.Start

   .. py:attribute:: Symbolic_Model
      :canonical: formak.ui_state_machine.StateId.Symbolic_Model
      :value: 'auto(...)'

      .. autodoc2-docstring:: formak.ui_state_machine.StateId.Symbolic_Model

   .. py:attribute:: Fit_Model
      :canonical: formak.ui_state_machine.StateId.Fit_Model
      :value: 'auto(...)'

      .. autodoc2-docstring:: formak.ui_state_machine.StateId.Fit_Model

.. py:class:: ConfigView(params: typing.Dict[str, typing.Any])
   :canonical: formak.ui_state_machine.ConfigView

   Bases: :py:obj:`formak.python.Config`

   .. autodoc2-docstring:: formak.ui_state_machine.ConfigView

   .. rubric:: Initialization

   .. autodoc2-docstring:: formak.ui_state_machine.ConfigView.__init__

   .. py:property:: common_subexpression_elimination
      :canonical: formak.ui_state_machine.ConfigView.common_subexpression_elimination
      :type: bool

      .. autodoc2-docstring:: formak.ui_state_machine.ConfigView.common_subexpression_elimination

   .. py:property:: python_modules
      :canonical: formak.ui_state_machine.ConfigView.python_modules

      .. autodoc2-docstring:: formak.ui_state_machine.ConfigView.python_modules

   .. py:property:: extra_validation
      :canonical: formak.ui_state_machine.ConfigView.extra_validation
      :type: bool

      .. autodoc2-docstring:: formak.ui_state_machine.ConfigView.extra_validation

   .. py:property:: max_dt_sec
      :canonical: formak.ui_state_machine.ConfigView.max_dt_sec
      :type: float

      .. autodoc2-docstring:: formak.ui_state_machine.ConfigView.max_dt_sec

   .. py:property:: innovation_filtering
      :canonical: formak.ui_state_machine.ConfigView.innovation_filtering
      :type: typing.Optional[float]

      .. autodoc2-docstring:: formak.ui_state_machine.ConfigView.innovation_filtering

.. py:class:: StateMachineState(name: str, history: typing.List[formak.ui_state_machine.StateId])
   :canonical: formak.ui_state_machine.StateMachineState

   .. autodoc2-docstring:: formak.ui_state_machine.StateMachineState

   .. rubric:: Initialization

   .. autodoc2-docstring:: formak.ui_state_machine.StateMachineState.__init__

   .. py:method:: state_id() -> formak.ui_state_machine.StateId
      :canonical: formak.ui_state_machine.StateMachineState.state_id
      :abstractmethod:
      :classmethod:

      .. autodoc2-docstring:: formak.ui_state_machine.StateMachineState.state_id

   .. py:method:: history() -> typing.List[formak.ui_state_machine.StateId]
      :canonical: formak.ui_state_machine.StateMachineState.history

      .. autodoc2-docstring:: formak.ui_state_machine.StateMachineState.history

   .. py:method:: available_transitions() -> typing.List[str]
      :canonical: formak.ui_state_machine.StateMachineState.available_transitions
      :abstractmethod:
      :classmethod:

      .. autodoc2-docstring:: formak.ui_state_machine.StateMachineState.available_transitions

   .. py:method:: search(end_state: formak.ui_state_machine.StateId, *, max_iter: int = 100, debug: bool = True) -> typing.List[str]
      :canonical: formak.ui_state_machine.StateMachineState.search

      .. autodoc2-docstring:: formak.ui_state_machine.StateMachineState.search

.. py:class:: NisScore
   :canonical: formak.ui_state_machine.NisScore

   .. autodoc2-docstring:: formak.ui_state_machine.NisScore

   .. py:method:: __call__(estimator: formak.python.SklearnEKFAdapter, X, y=None) -> float
      :canonical: formak.ui_state_machine.NisScore.__call__

      .. autodoc2-docstring:: formak.ui_state_machine.NisScore.__call__

.. py:data:: PIPELINE_STAGE_NAME
   :canonical: formak.ui_state_machine.PIPELINE_STAGE_NAME
   :value: 'kalman'

   .. autodoc2-docstring:: formak.ui_state_machine.PIPELINE_STAGE_NAME

.. py:class:: FitModelState(name: str, history: typing.List[formak.ui_state_machine.StateId], model: formak.ui_model.Model, parameter_space: typing.Dict[str, typing.List[typing.Any]], parameter_sampling_strategy, data, cross_validation_strategy)
   :canonical: formak.ui_state_machine.FitModelState

   Bases: :py:obj:`formak.ui_state_machine.StateMachineState`

   .. autodoc2-docstring:: formak.ui_state_machine.FitModelState

   .. rubric:: Initialization

   .. autodoc2-docstring:: formak.ui_state_machine.FitModelState.__init__

   .. py:method:: state_id() -> formak.ui_state_machine.StateId
      :canonical: formak.ui_state_machine.FitModelState.state_id
      :classmethod:

      .. autodoc2-docstring:: formak.ui_state_machine.FitModelState.state_id

   .. py:method:: available_transitions() -> typing.List[str]
      :canonical: formak.ui_state_machine.FitModelState.available_transitions
      :classmethod:

   .. py:method:: export_python() -> formak.python.ExtendedKalmanFilter
      :canonical: formak.ui_state_machine.FitModelState.export_python

      .. autodoc2-docstring:: formak.ui_state_machine.FitModelState.export_python

   .. py:method:: _fit_model_impl(debug_print=False)
      :canonical: formak.ui_state_machine.FitModelState._fit_model_impl

      .. autodoc2-docstring:: formak.ui_state_machine.FitModelState._fit_model_impl

.. py:class:: SymbolicModelState(name: str, history: typing.List[formak.ui_state_machine.StateId], model: formak.ui_model.Model)
   :canonical: formak.ui_state_machine.SymbolicModelState

   Bases: :py:obj:`formak.ui_state_machine.StateMachineState`

   .. autodoc2-docstring:: formak.ui_state_machine.SymbolicModelState

   .. rubric:: Initialization

   .. autodoc2-docstring:: formak.ui_state_machine.SymbolicModelState.__init__

   .. py:method:: state_id() -> formak.ui_state_machine.StateId
      :canonical: formak.ui_state_machine.SymbolicModelState.state_id
      :classmethod:

      .. autodoc2-docstring:: formak.ui_state_machine.SymbolicModelState.state_id

   .. py:method:: available_transitions() -> typing.List[str]
      :canonical: formak.ui_state_machine.SymbolicModelState.available_transitions
      :classmethod:

   .. py:method:: fit_model(parameter_space: typing.Dict[str, typing.List[typing.Any]], data, *, parameter_sampling_strategy=None, cross_validation_strategy=None) -> formak.ui_state_machine.FitModelState
      :canonical: formak.ui_state_machine.SymbolicModelState.fit_model

      .. autodoc2-docstring:: formak.ui_state_machine.SymbolicModelState.fit_model

.. py:class:: DesignManager(name)
   :canonical: formak.ui_state_machine.DesignManager

   Bases: :py:obj:`formak.ui_state_machine.StateMachineState`

   .. autodoc2-docstring:: formak.ui_state_machine.DesignManager

   .. rubric:: Initialization

   .. autodoc2-docstring:: formak.ui_state_machine.DesignManager.__init__

   .. py:method:: state_id() -> formak.ui_state_machine.StateId
      :canonical: formak.ui_state_machine.DesignManager.state_id
      :classmethod:

      .. autodoc2-docstring:: formak.ui_state_machine.DesignManager.state_id

   .. py:method:: available_transitions() -> typing.List[str]
      :canonical: formak.ui_state_machine.DesignManager.available_transitions
      :classmethod:

   .. py:method:: symbolic_model(model: formak.ui_model.Model) -> formak.ui_state_machine.SymbolicModelState
      :canonical: formak.ui_state_machine.DesignManager.symbolic_model

      .. autodoc2-docstring:: formak.ui_state_machine.DesignManager.symbolic_model
