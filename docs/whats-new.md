# What's New

Release Notes

## 2023-08-04 Introducing FormaK Runtime

Commit [`commit`]()

The first piece of the FormaK Runtime is the `ManagedFilter`. This will
coordinate new sensor readings, process model updates and sensor model updates
with an easy to use interface.

The FormaK runtime will continue to be expanded over time to offer new features
(e.g. around netcode and better managing sensor readings) as well as offering
additional opt-in services (e.g. logging).

Other Improvements:
- Reorganize how the AST is packaged with reusable fragments
- Fix misconfigured CI
- Fix misconfigured Linting

## 2023-06-29 Common Subexpression Elimination

Commit [`80fdd9d`](https://github.com/buckbaskin/formak/commit/80fdd9dca89b3f15b53baed0ebea56ead8a7f432)

Common Subexpression Elimination removes all unnecessary code from the compute
graph for the process model and individual sensor models in the filter. This
represents a straightforward performance win that should benefit all models
without changing the compute outcome. The upside for enabling the feature grows
with model size because Common Subexpression Elimination with reduce shared
computation at all scales, both large segments of repeat computation and the
smallest operations.

## 2023-05-05 Calibration

Commit [`02005ce`](https://github.com/buckbaskin/formak/commit/02005ce4fe932f5ad4d1131b117fa0b0a20232b9)

Add an optional Calibration parameter for setting up multiple sensor models or
adding sensors at known positions.

Other Improvements:
- Performance for the `ui.Model` and code generation flows is much improved

Calibration Design: [designs/calibration.md](../designs/calibration.md)
Getting Started: [getting-started.md](getting-started.md)

## 2023-04-10 C++ Source for Models

Commit [`d2bec5c`](https://github.com/buckbaskin/formak/commit/d2bec5c7ea27f8092ea6d28c61917e7926fb8e72)

Generate C++ source for models as an add on to the symbolic and Python sources for models

## 2022-12-21 Scikit-Learn Integration

Commit [`ecbdb9e`](https://github.com/buckbaskin/formak/commit/ecbdb9ecf4812cdd12b0fc5194e23ebed6718978)

FormaK Python models adapt the interface for Scikit-Learn (train, test, etc).
This enables easier model fitting and additional integration with the Python
science tools ecosystem.

## 2022-09-13 Python Source for Models

Commit [`7e5ba82`](https://github.com/buckbaskin/formak/commit/7e5ba82c2c7bc0307bd145cc7a7c5d55c3e917f2)

Python classes for models and Kalman filters

## 2022-08-26 Python User Interface

Commit [`578ea7e`](https://github.com/buckbaskin/formak/commit/578ea7e721637ce3a2a16768b8bba49d4dd94130)

Demonstrate the User Interface for FormaK
