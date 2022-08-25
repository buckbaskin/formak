# FormaK V0

Author: Buck Baskin @bebaskin

## Overview

FormaK aims to combine symbolic modeling for fast, efficient system modelling
with code generation to create performant code that is easy to use.

The values (in order) are:
- Easy to use
- Performant

In line with those values, the intended user experience is as follows. The user
provides:
- Model that describes the physics of the system
- Execution criteria (e.g. memory usage, execution time)
- Time series data for for the system

FormaK provides the tooling so that you can easily derive Python and C++
implementations that are as fast as possible to best match your data given the
constraints.

### The Five Keys

The Five Key Elements the library provides to achieve this are:
1. Python Interface to define models
2. Python implementation of the model and supporting tooling
3. Integration to scikit-learn to leverage the model selection and parameter
   tuning functions
4. C++ and Python to C++ interoperability for performance
5. C++ interfaces to support a variety of model uses

### Under the Hood

Under the hood, the library will provide optimizations for improved accuracy,
selecting better underlying problem formulations and reduced compute time during
model execution. The library will also perform transparent optimizations at the
model-definition and C++ implementation levels.

## Roadmap and Milestones

- [ ] Python UI Demo - defining models
- [ ] Python Models
- [ ] Python Model Selection with scikit-learn integration
- [ ] C++ Generation Demo
- [ ] Automated CI
- [ ] Automated Feature Test Reporting
- [ ] Writeup of the demo

## Development Process

The development process for adding features will be:

1. Write a Design Doc
2. Write feature tests. When the feature tests pass, the feature is nominally
working at an alpha level
3. Build a simple prototype
4. Implement the feature, including additional testing
5. Code Review, Refactor
6. Merge via PR
7. Write up successes, retro of what changed (to include that feedback in future
   designs)

Note 1: This project covers three "languages":
- Python
- C++
- C++ wrapped in Python

This means that most features will have feature tests across the three language
use cases.

Note 2: Features can track and share their status by referring to these stages
in the development process. This makes it easier to share what features are in
what stage of the development process over time and for each release.

### Example Development Process

Making this a little more specific, the development process for calling C++ from
Python would look something like the following:

1. Write a Design Doc. This includes context for the feature, the problem (use
   C++ from Python), the expected solution approach and feature tests. The
   feature tests function as success criteria for the design
2. Write the feature test(s): Python file with tests where the tests pass if it
   successfully calls C++ logic
3. Build a simple prototype: call a simple C++ function that returns integers
4. Build the feature to wrap the C++ model library interface
5. Add additional unit tests to explore variations of possible model uses
6. Code review
7. Merge PR
8. Write up the project

## Solution Approach

Combine tooling and libraries in Python:
- sympy
- scikit-learn
- hypothesis
- etc

with libraries and benefits of C++:
- performant, only pay for what you use
- Eigen
- etc

and end up with an easy to use interface so it's easy to adopt and use.  Working
together, we can build "Middle-Out". On the frontend, we can add features to the
Python interface. On the backend, we augment the features with performant
implementations.

- Python, and specifically sympy, is the primary interface for interacting with
  the library with a focus on ease of use.
- C++ is a "backend" that executes the model defined in Python with a focus on
  performance.
- Pure Python and Python wrapping C++ are alternative backends for executing the
  model

