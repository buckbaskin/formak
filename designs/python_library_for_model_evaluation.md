# Python Library for Model Evaluation

Author: Buck Baskin @bebaskin
Created: 2022-08-26
Updated: 2022-08-26
Parent Design: [designs/formak_v0.md](../designs/formak_v0.md)
See Also: [designs/python_ui_demo.md](../designs/python_ui_demo.md)
Status: 5. Refactor

## Overview

FormaK aims to combine symbolic modeling for fast, efficient system modelling
with code generation to create performant code that is easy to use.

The values (in order) are:

- Easy to use
- Performant

The Five Key Elements the library provides to achieve this (see parent) are:
1. Python Interface to define models
2. Python implementation of the model and supporting tooling
3. Integration to scikit-learn to leverage the model selection and parameter tuning functions
4. C++ and Python to C++ interoperability for performance
5. C++ interfaces to support a variety of model uses

This design provides the initial implementation of second of the Five Keys
"Python implementation of the model and supporting tooling". This design also
prepares for the third of the 5 Keys: "Integration to scikit-learn to leverage
the model selection and parameter tuning functions". At this stage it is
helpful to inform the design of the tooling so that it won't have any big
hurdles to the next steps in the design.

## Solution Approach

The basic step will be to translate from Sympy to Python (without a sympy
dependency). Sympy provides this functionality already, so getting the basics
working won't be too hard. The follow on work to refactor will be important in
order to make sure that the library remains easy to use.

The key classes involved are:
- `ui.Model`: User interface class encapsulating the information required to
  define the model
- `py.Model`: (new) Class encapsulating the model for running a model
  efficiently in Python code

To keep things neatly separated, the translation from `ui.Model` to `py.Model`
will be handled by a separate free function that takes a `ui.Model` as an
argument and returns a `py.Model`.

### Tooling

Along with the `py.Model` encapsulation, also provide an Extended Kalman Filter
implementation to quantify variance (based on best fit of a Kalman Filter to
data) and outliers (innovation as a function of variance). This part of the
design is more focused on being used with the coming scikit-learn integration.

The key classes involved are:
- `py.Model`: (new) Class encapsulating the model for running a model efficiently in Python code
- `py.ExtendedKalmanFilter`: (new) 
	- Looking ahead to model fitting, characterize model quality, data variance by fitting an EKF
	- Constructor should accept state type, state to state process model (`py.Model`? `ui.Model`?), process noise, sensor types, state to sensor sensor models, sensor noise
	- Process Model Function: take in current state, current variance, dt/update time. Return new state, new variance
	- Sensor Model Function: take in current state, current variance, sensor id, sensor reading

These two classes will likely share a lot under the hood because they both want
to run Python efficiently; however, they'll remain independent classes to start
for a separation of concerns. The EKF class at this point is more aimed to
using it under the hood of the scikit-learn stuff whereas the `py.Model` class
is aimed at the Formak User (easy to use first, performant second).

Notes:
- Numpy will likely feature heavily here

### The Cherry On Top - Transparent Compilation

In addition to merely repackaging the model defined in the `ui.Model`, this
design will integrate with Python compiler tooling (something like
Numba/Cython) to write Python in the `py.Model` class, but JIT compile or
C-Compile high use model functions.

This will have some trade-offs (increased implementation complexity, increased
startup time), but should likely also have some performance benefits especially
for longer-running analysis use cases (e.g. running with a long sequence of
data).

Notes:
- Don't forget the order of the values: easy to use first, performant second. The compiler shouldn't unnecessarily complicate the interface to the `py.Model` class
- The particular compiler will be selected during the project by experimenting with different compilers
- In the spirit of don't pay for what you don't use, this will also motivate the creation of a common configuration pattern. We want to be able to (at conversion time) selectively enable or disable the compilation. Putting some thought into a common configuration pattern will make it easier to reuse in future designs (e.g. selecting configuration about other model optimizations)
- The configuration should also be able to be disabled automatically if the selected compiler library isn't available. This will ensure that the dependency on the compiler is optional (but recommended).

The Python compiler step will require some basic profiling as a feature
test/acceptance test.

## Feature Tests

This feature is specific to the Python interface. There will be four feature
tests:
1. UI -> Python: Simple 2D model of a parabolic trajectory converting from `ui.Model` to `py.Model` (no compilation)
2. Tooling: Simple 2D model of a parabolic trajectory converting from `ui.Model` to `py.ExtendedKalmanFilter`
3. Compilation: Simple 2D model of a parabolic trajectory converting from `ui.Model` to `py.Model` (compilation)
4. Compilation: Model converting from `ui.Model` to `py.Model`. Run against a sequence of data and profile. Assert `py.Model` with compilation faster than no compilation (excluding startup time).

For the compilation specifically, if there aren't any performance benefits to
be demonstrated, then remove it from the PR  in favor of a later design that
can more specifically focus on compilation.

## Road Map and Process

1. Write a design
2. Write a feature test(s)
3. Build a simple prototype
4. Pass feature tests
5. Refactor/cleanup
6. Build an instructive prototype (e.g. something that looks like the project vision but doesn’t need to be the full thing)
7. Add unit testing, etc
8. Refactor/cleanup
9. Write up successes, retro of what changed (so I can check for this in future designs)

## Post Review

### 2022-09-10

Selecting `numba` for the Python compiler
- Pure python written source code
- Simple Python -> compiled Python syntax
- Designed for use with numpy
- SIMD vectorization under the hood
