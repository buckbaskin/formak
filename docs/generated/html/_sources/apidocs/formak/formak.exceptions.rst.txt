:py:mod:`formak.exceptions`
===========================

.. py:module:: formak.exceptions

.. autodoc2-docstring:: formak.exceptions
   :allowtitles:

Module Contents
---------------

API
~~~

.. py:exception:: FormakBaseException()
   :canonical: formak.exceptions.FormakBaseException

   Bases: :py:obj:`Exception`

.. py:exception:: MinimizationFailure(minimization_result, *args, **kwargs)
   :canonical: formak.exceptions.MinimizationFailure

   Bases: :py:obj:`formak.exceptions.FormakBaseException`

   .. py:method:: __str__()
      :canonical: formak.exceptions.MinimizationFailure.__str__

   .. py:method:: __repr__()
      :canonical: formak.exceptions.MinimizationFailure.__repr__

.. py:exception:: ModelDefinitionError()
   :canonical: formak.exceptions.ModelDefinitionError

   Bases: :py:obj:`formak.exceptions.FormakBaseException`

.. py:exception:: ModelConstructionError()
   :canonical: formak.exceptions.ModelConstructionError

   Bases: :py:obj:`formak.exceptions.FormakBaseException`

.. py:exception:: ModelFitError()
   :canonical: formak.exceptions.ModelFitError

   Bases: :py:obj:`formak.exceptions.FormakBaseException`
