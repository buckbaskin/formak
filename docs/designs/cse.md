# Common Subexpression Elimination

Author: Buck Baskin @bebaskin
Created: 2023-06-15
Updated: 2023-06-25
Parent Design: [designs/cpp_library_for_model_evaluation.md](../designs/cpp_library_for_model_evaluation.md)
See Also: [designs/python_library_for_model_evaluation.md](../designs/python_library_for_model_evaluation.md)
Status: Refactor

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
to support a variety of model uses" to support the performance goals of the
library: Common Subexpresion Elimination (CSE).

CSE is a transformation on the compute graph of the underlying model (for
either Python or C++) to remove duplicate compuation at multiple scales. For
example, if a sensor model computes `a + b` multiple times, CSE identifies this
subexpression and computes it once. This could also apply to a more complicated
expression like `9.81 * sin(theta + bias)`.

One of the nicest benefits of this transformation is that it will provide a
performance benefit to the model without a compromise to the user interface.

## Solution Approach

The basic approach will be to apply `sympy`'s common subexpression elimination
implementation to the sympy model before generating the output. This will be an
overhaul of both the Python and C++ implementations, but for different reasons.
The C++ implementation doesn't do any CSE now. The Python implementation is
more subtle. Currently the CSE algorithm is applied for each field of the
sensor model, but the correct full process is to eliminate subexpressions
across all fields. This should offer improved performance (and at least no
regression in performance).

The key classes in the implementation are:
- Basic Block: Basic Block wraps uninterrupted sequences of assignments, so CSE can be applied across all assignments in the CSE (as long as the output is ordered so all sub-expressions are computed before they are used). The Python implementation may also adopt the Basic Block pattern.
- Config: Adding/Updating a new feature flag for CSE

## Feature Tests

The feature tests for this design are based on generating models with many
common subexpressions, then comparing the performance of the model with and
without CSE.

- Generate a tree of `2 * sin(l * r) * cos(l * r)`
- At its leaves, `l` and `r` are input symbols
- At the inner nodes, `l` and `r` are left and right subexpressions in the above pattern

In each feature test, the time to execute should grown in log(N) where N is the
number of nodes for the CSE implementation; however, without CSE it should grow
proportional to N. This should be a 10x performance improvement for between 30
and 40 nodes. In practice it may take more nodes to demonstrate a difference if
the compute time is small for all cases.

In unit testing, the difference can be more precisely tested by generating a
common subexpression then asserting it gets removed in the output.

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
