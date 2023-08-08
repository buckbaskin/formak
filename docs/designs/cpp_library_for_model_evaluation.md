# C++ Library for Model Evaluation

Author: Buck Baskin @buck@fosstodon.org
Created: 2023-01-08
Updated: 2023-04-05
Parent Design: [designs/formak_v0.md](../designs/formak_v0.md)
See Also: [designs/python_library_for_model_evaluation.md](../designs/python_library_for_model_evaluation.md)
Status: Merged

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

This design provides the initial implementation of fifth of the Five Keys
"C++ interfaces to support a variety of model uses".

## Solution Approach

The basic step will be to translate from Sympy to C++. Sympy provides this
functionality as one of two systems: code printers and code generators. To
enable additional customization, the initial implementation will use the code
printers with templating instead of the code generators (which provide
additional functionality at the expense of additional constraints).

The follow on work to refactor will be important in
order to make sure that the library remains easy to use. This will include
cleaning up the Python and C++ templates as well as using a Bazel macro to make
the C++ generation a unified rule instead of hand-rolling multiple rules.

The key classes in the implementation are:
- `ui.Model`: User interface class encapsulating the information required to
  define the model
- `cpp.Model`: (new) Class encapsulating the model for generating a model
  in C++

The key output classes will be:
- `class Model`: C++ header and source file corresponding to the implementation of the model. Generated with a namespace and name customization

### Tooling

Along with the `class Model` implementation, also provide an Extended Kalman Filter
implementation to quantify variance (based on best fit of a Kalman Filter to
data) and outliers (innovation as a function of variance).

The key classes involved are:
- `cpp.Model`: (new) Class encapsulating the model for running a model efficiently in C++
- `cpp.ExtendedKalmanFilter`: (new)
	- Constructor should accept state type, state to state process model (`py.Model`? `ui.Model`?), process noise, sensor types, state to sensor models, sensor noise
	- Process Model Function: take in current state, current variance, dt/update time. Return new state, new variance
	- Sensor Model Function: take in current state, current variance, sensor id, sensor reading

These two classes will likely share a lot under the hood because they both want
to run C++ efficiently; however, they'll remain independent classes to start
for a separation of concerns. These two classes will also share an interface
with the Python implementation as much as is reasonable to provide easier
interopoeration between the two languages (for Key Element #4)

### Sympy

Key Features used from Sympy that should translate across both Python and C++ implementations:
- [lambdify](https://docs.sympy.org/latest/modules/utilities/lambdify.html#sympy.utilities.lambdify.lambdify)
- [common subexpression elimination](https://docs.sympy.org/latest/modules/simplify/simplify.html#sympy.simplify.cse_main.cse)
- [jacobian](https://docs.sympy.org/latest/modules/matrices/matrices.html#sympy.matrices.matrices.MatrixCalculus.jacobian)

### Math

Following the EKF math from Probabilistic Robotics
- S. Thrun, W. Burgard, and D. Fox, Probabilistic robotics. Cambridge, Mass.: Mit Press, 2010.


## Feature Tests

This feature is specific to the C++ interface. There will be two feature tests:
1. UI -> C++: Simple 2D model of a parabolic trajectory converting from `ui.Model` to `cpp.Model`
2. Tooling: Simple 2D model of a parabolic trajectory converting from `ui.Model` to `cpp.ExtendedKalmanFilter`

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

### 2023-04-05

This deisgn took way longer to implement than I'd hoped. I'm going to instead
aim for designs that should take about a month and then review after the fact.
In this case, I'm off by a factor of 3...

#### Design Changes - Code

- Complete rewrite of C++ interface for EKF
- Implementation patterns for EKF
- Iterated on multiple code patterns for generating C++
- Long list of TODOs for internal improvements but shipping for now
- C++ stats header
- Not using common subexpression elimination yet for C++ generation
- Skipped EKF math for probability of each reading
- Added new checks for "model collapse" to zero covariance with sympy solve (nonlinsolve)
- Changed of process noise definition to match process definition with keys instead of indexing
- Refactored for common functions across py/cpp, model/EKF
- Spent lots of time making generated whitespace look nicer
- Never found a satisfying mix of Jinja template vs Python codegen

#### Design Changes - Tooling

- docs diff for Github Actions
- precommit as a tool
- bazel rule to automate C++ generation
- py modernize tooling
	- format strings
	- yield from

#### Some Things I Learned I Didn't Know

- C++ toolchains in bazel
- Managing Docker containers and cleanup doing edit-run-observe-kill loop
- clang-tidy in bazel
- Mental model of bazel
