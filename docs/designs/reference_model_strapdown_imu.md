# Innovation Filtering

Author: Buck Baskin @buck@fosstodon.org
Created: 2023-09-19
Updated: 2023-09-19
Parent Design: [designs/python_ui_demo.md](../designs/python_ui_demo.md)


## Overview

FormaK aims to combine symbolic modeling for fast, efficient system modelling
with code generation to create performant code that is easy to use.

The Five Key Elements the library provides to achieve this user experience are:
1. Python Interface to define models
2. Python implementation of the model and supporting tooling
3. Integration to scikit-learn to leverage the model selection and parameter tuning functions
4. C++ and Python to C++ interoperability for performance
5. C++ interfaces to support a variety of model uses

This design focuses on an example of the "Python Interface to define models" to define a reference model for a strapdown IMU. This serves two purposes:
1. Provide a reference for implementing a strapdown IMU as a part of other models
2. Further exercise the FormaK interface

As a third consideration, this will also provide a reference design for how
other reference models will be designed and presented in the future.

## Solution Approach

### The Strapdown IMU

What is a strapdown IMU?

IMU mounted to the vehicle of interest

### Definitions

This design will implement the strapdown IMU model defined by the source
[Strapdown inertial navigation | Rotations](https://rotations.berkeley.edu/strapdown-inertial-navigation/).
The site is a joint resource from the mechanical engineering departments at
Rose-Hulman Institute of Technology and UC Berkeley.

![Definition of terminology and axis](assets/reference_model_strapdown_imu/tracked-body.png)
[Source](https://rotations.berkeley.edu/wp-content/uploads/2017/10/tracked-body.png)

- $e_{i}$ axis of rigid body (1, 2, 3)
- $\omega$ vector of rotations of the rigid body
- $\omega_{i}(t) = \omega \cdot e_{i}$ IMU reading of rotation
- $g$ acceleration due to gravity
- $\ddot{x_{A}}$ acceleration of the rigid body at the IMU measurement point A
- $f_{i}(t) = (\ddot{x_{A}} - g) \cdot e_{i}$ IMU reading of acceleration (specific force)

The reference design uses 3-2-1 Euler angles.

TODO(buck): embed these visualizations and finish explanation

![Rotations](https://rotations.berkeley.edu/wp-content/ql-cache/quicklatex.com-5bc0ef31513d8f6aa027b50b28f7dba9_l3.svg)

![Accelerations](https://rotations.berkeley.edu/wp-content/ql-cache/quicklatex.com-7dc4cf09b3717d6ebc1d7ca32a1e3dda_l3.svg)

## Feature Tests

The "Rotations" resource also provides an implementation of the strapdown IMU
model for tracking a
[tumbling smartphone](https://rotations.berkeley.edu/reconstructing-the-motion-of-a-tossed-iphone/).

The feature test will implement the strapdown model based on the data provided
and revisions to the reference model suggested in the resource.

## Roadmap and Process

1. Write a design
2. Write a feature test(s)
3A. Experiments
3B. Build a simple prototype
4. Pass feature tests
5. Refactor/cleanup
6. Build an instructive prototype (e.g. something that looks like the project vision but doesn't need to be the full thing)
7. Add unit testing, etc
8. Refactor/cleanup
9. Write up successes, retro of what changed (so I can check for this in future designs)

## Post Review
