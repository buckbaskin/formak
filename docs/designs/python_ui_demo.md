# Python UI Demo

Author: Buck Baskin @bebaskin

Parent Design: [designs/formak_v0.md](../designs/formak_v0.md)
Status: Merged

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

This design provides the first of the Five Keys: the Python Interface to define models.

## Roadmap and Milestones

The development process for the feature will be:

1. Design Doc
2. Write feature tests. When the feature tests pass, the feature is nominally working at an alpha level
3. Build a simple prototype
4. Implement the feature, including additional testing
5. Code Review, Refactor
6. Merge via PR
7. Write up successes, retro of what changed (to include that feedback in future designs)

## Solution Approach

To start, the user interface will lean on the
[`sympy`](https://www.sympy.org/en/index.html) package for symbolic math. Sympy
shares the value of being easy to use. In addtion, leaning on sympy instead of a
proprietary interface enables a lot of flexibility and future progress for
things like code generation based on the model.

Likely it'll make sense to wrap common sympy operations that are specific and
repeated in this project, but this kind of tooling will be developed in flight
based on pain points testing with example models.

## Feature Tests

This feature is specific to the Python interface. There will be two feature tests:
1. Simple 2D model of a parabolic trajectory
2. Rocket launch model with thrust, drag and other effects

## Post Merge Review

This feature was ultimately very simple because it leaned on `sympy` quite a
bit. The primary change from the plan was that there was already demo code to
borrow from for feature tests which made writing them much quicker.
