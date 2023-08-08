# Calibration

Author: Buck Baskin @buck@fosstodon.org
Created: 2023-05-06
Updated: 2023-05-06
Parent Design: [designs/cpp_library_for_model_evaluation.md](../designs/cpp_library_for_model_evaluation.md)
See Also: [designs/python_library_for_model_evaluation.md](../designs/python_library_for_model_evaluation.md)
Status: Merged

## Overview

FormaK aims to combine symbolic modeling for fast, efficient system modelling
with code generation to create performant code that is easy to use.

This design provides an extension to the fifth of the Five Keys
"C++ interfaces to support a variety of model uses".

When defining a model, in theory everything can be estimated as a state or
provided as a control input; however, having the ability to provide
calibrations can be helpful or even essential. For example, in the NASA rocket
dataset the position of the IMU on the rocket is calibrated ahead of time so it
doesn't need to be estimated online.

For the case of a suite of ranging sensors (say ultra-wideband time of flight
sensors), the calibration term allows for easily setting up a single model with
different calibrations for the known positions of each sensor in the reference
frame. Without the calibration, each pose would be arbitrary and require
solving a problem beyond what is suited to a Kalman Filter. With the
calibration, the sensor model can be defined once and then calibrated multiple
times at runtime based on how the sensors are set up for a particular use case.

The Calibration use case also provides additional functionality on top of the
Control inputs. The two categories conceptually overlap as "known" values that
are accepted in the state update; however, the Calibration values are also
available within the sensor model. With just a State and Control input, the
state needs to accept control inputs as a pass through to sensor models. This
adds a compute penalty for computations with the state.

Supporting calibration is a big step forward in functionality for FormaK that
enables a variety of new model use cases.

## Solution Approach

The basic approach will be to pass in calibrated values to both the process
model and sensor model, largely following the implementation of the Control
type (except that it will also be provided to sensor models).

The key classes in the implementation are:
- `ui.Model`: Revised to support the calibration parameters
- `python.Model`: Revised to support the calibration parameters
- Generated `Calibration`: (new) Generated type to provide calibration terms at runtime

## Feature Tests

This design was retcon'd based on a feature test of developing a model for a
rocket launch based on NASA data.

[NASA Dataset Page](https://data.nasa.gov/Aerospace/Deorbit-Descent-and-Landing-Flight-1-DDL-F1-/vicw-ivgd)
[YouTube Video of Launch](https://www.youtube.com/watch?v=O97dPDkUGg4)

The feature tests for this design are based on defining a model of the rocket
motion and then generating Python and C++ models for the model. The
implementation exposed a missing aspect of the FormaK model, specifically the
introduction of the information about the pose of the IMU in the navigation
frame of the rocket. This is provided with the dataset, but is not easily
integrated into the model when it could only be a state (and therefore
estimated based on an initial condition) or control (and therefore not
available when calculating a sensor model for the IMU).

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

### 2023-05-05

This design (cut out of the broader rocket model work) accidentally took
exactly how long I wanted, specifically landing one month after the post-review
of the previous feature. This was a happy accident because I'd originally
intended for the rocket model itself to land in this time and instead wandered
into this feature to design and deliver early.

#### Original List of Features Expected For The Rocket Model

Note: the only thing that came out of this was the calibration, partial vector
support and partial rotation support...

- Vector support
- Rotations
- Units
- Map
- Data ingestion helpers

#### Design Changes - Code

- Passing calibration into every function call. In the future, a managed filter can construct the calibration once, but for now they're stateless "pure" functions so they need to get the calibration passed in.
- Update of the interface to be "don't pay for what you don't use". This applies both to optional arguments on the Python side and C++ that is conditionally generated only if the appropriate Calibration/Control is needed
- Rotations, translations, velocities, etc got their own named generators in the model definition code. I expect this will be expanded in the future to enable easier model generation and moved into the UI code itself (e.g. rigid transforms, etc)
- Overall, I opted to remove some of the `ui.Model` functionality that was taking a long time for a larger model in favor of faster iteration and some testing after the fact. This was a key win because I was sitting around for 5 minutes at a time at the slowest point
- Better error messages along the way. I had enough failures and time to think to find the failure, write an error message for it and rewrite the error message the second time around

#### Design Changes - Tooling

- Complete rewrite of C++ code gen templates with if-based optional inclusion. This got quite messy and is still in Jinja but maybe not for long.
- I chose to unwind some of the changes I'd made to check models for invalid cases. It was slow to execute and false-positive prone.
- Basic testing went a long way to finding obvious stuff and not-so obvious stuff. I bet there are edge cases, but most of the basics are covered
- Sympy `simplify` is too slow to be useful without a more careful application
- It's helpful to not have to write to a file all the time. Tests will just dump the model writing to stdout if there's no file specified so the C++ compile calls can be run in tests

#### Some Things I Learned I Didn't Know

- Rotations. I have the nominal math, but still not a completely satisfying approach
- Benchmarking is important even for smaller models
