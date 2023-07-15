# FormaK Managed Runtime

Author: Buck Baskin @bebaskin
Created: 2023-07-13
Updated: 2023-07-13
Parent Design: [designs/cpp_library_for_model_evaluation.md](../designs/cpp_library_for_model_evaluation.md)
See Also: [designs/python_library_for_model_evaluation.md](../designs/python_library_for_model_evaluation.md)
Status: Design

## Overview

FormaK aims to combine symbolic modeling for fast, efficient system modelling
with code generation to create performant code that is easy to use.

The Five Key Elements the library provides to achieve this user experience are:
1. Python Interface to define models
2. Python implementation of the model and supporting tooling
3. Integration to scikit-learn to leverage the model selection and parameter tuning functions
4. C++ and Python to C++ interoperability for performance
5. C++ interfaces to support a variety of model uses

This design provides an extension to the second of the Five Keys "Python
implementation of the model..." and the fifth of the Five Keys "C++ interfaces
to support a variety of model uses" to support easier to use filters. The
current implementation provides some of the math for the EKF (process update,
sensor update) which can be used in a flexible manner, but don't necessarily
meet the easy to use goal for the project.

This feature aims to make the library easier to use by providing a managed
filter with a single interface.

This feature also has a forward looking benefit: by setting up the Managed
Filter structure now, it will be easier to add "netcode" features, logging and
other runtime benefits within the context of a unified Managed Filter.

## Solution Approach

The primary class of interest will be a new class `ManagedFilter` (in both C++
and Python).

This class will have a member function `tick` that will be the primary user
facing to the rest of the filter logic.

    // No sensor reading, process update only
    tick(time)
    tick(time, [])

    // One sensor reading
    tick(time, [reading])

    // Multiple sensor readings
    tick(time, [reading1, reading2, reading3])

The function will return the current state and variance of the filter after
processing the tick. By using the `ManagedFilter` wrapper, the user doesn't
need to worry about tracking states, variances, times, the process model (or
models in the future), sensor models. Instead the user just passes in the
desired output time and any new information from sensors and gets the result.

## Feature Tests

The feature tests for this design will focus on the tick interface in a few
combinations:

- No sensor updates (process only)
- One sensor update
- Multiple sensor updates

The goal is to focus on the filter management and not the EKF math itself, so
assertions will focus on time management and broad trends in state and variance
for a simple model where it's easy to calculate model evolution by hand.

## Road Map and Process

1. Write a design
2. Write a feature test(s)
3. Build a simple prototype
4. Pass feature tests
5. Refactor/cleanup
6. Build an instructive prototype (e.g. something that looks like the project vision but doesn't need to be the full thing)
7. Add unit testing, etc
8. Refactor/cleanup
9. Write up successes, retro of what changed (so I can check for this in future designs)

## Post Review
