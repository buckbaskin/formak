# Hyper-parameter Selection

Author: Buck Baskin [@buck@fosstodon.org](https://fosstodon.org/@buck)
Created: 2023-11-08
Updated: 2023-11-08
Parent Design: [designs/sklearn-integration.md](../designs/sklearn-integration.md)
See Also: [designs/python_library_for_model_evaluation.md](../designs/python_library_for_model_evaluation.md), [designs/innovation_filtering.md](../designs/innovation_filtering.md)


## Overview

FormaK aims to combine symbolic modeling for fast, efficient system modelling
with code generation to create performant code that is easy to use.

The Five Key Elements the library provides to achieve this user experience are:
1. Python Interface to define models
2. Python implementation of the model and supporting tooling
3. Integration to scikit-learn to leverage the model selection and parameter tuning functions
4. C++ and Python to C++ interoperability for performance
5. C++ interfaces to support a variety of model uses

This design focuses on "Integration to scikit-learn to leverage the model
selection and parameter tuning functions". More specifically, this design
focuses on using scikit-learn tooling to automatically select the innovation
filtering level from data.

The promise of this design is that all parameters could be selected
automatically based on data instead of requiring hand tuning; however, this
design will focus narrowly on selecting the innovation filtering level as a
motivating example.

## Solution Approach

How can FormaK build a repeatable process for selecting model parameters?

To start, FormaK will follow the process laid out by [scikit-learn docs for tuning the hyper-parameters of an estimator](https://scikit-learn.org/stable/modules/grid_search.html#tuning-the-hyper-parameters-of-an-estimator)

The high level process is composed of the following elements:
- The estimator
- The parameter space
- The method for searching or sampling candidates
- The cross-validation scheme
- The score function

### Estimator

FormaK is already equipped to define symbolic models and then generate Python
estimators that follow the scikit-learn interface. While I don't expect this
aspect to change significantly, if a major interface change will make the
Python estimators easier to use then I am open to reworking the interface
because I suspect making this design easier to implement is correlated with
making the classes easier to use and easier to use correctly.

#### Estimator Interface

The design will have the user provide the symbolic model and allow the FormaK
tooling to handle generating a Python implementation.

### Parameter Space

The parameter space for this problem is positive floats. The problem could be
thought of as positive integers (how many standard deviations are acceptable)
but the positive floats should be easier to optimize as a continuous parameter.

The default method for parameter optimization is
[exhaustive grid search](https://scikit-learn.org/stable/modules/grid_search.html#exhaustive-grid-search).
For this problem of selecting a single or reduced set of parameters it should
be okay to use the baseline; however, the design will include an experiment to
compare between grid search and randomized search. The benefit of randomized
search is that a specific budget can be set and the approach will, with high
probability, find the best result in the same search space as the grid search
but with fewer iterations.

The parameter search algorithm will be an optional configuration option, with
FormaK picking a sane default (likely random search).

### Cross Validation

The
[cross-validation approach](https://scikit-learn.org/stable/modules/cross_validation.html)
suggested by scikit-learn to choose the hyper-parameter with the highest
cross-validation score:
1. Hold aside a test time series that does not overlap with the rest of the data set
2. Within the remaining data, the training time series, subdivide into different folds where one part of each fold is used to train the model and a subset is used for validation. By selecting many folds, we can reduce the chance that the parameters are overfit to noise in a particular test evaluation.
3. Perform final scoring against the held out test data.

One thing to note: I've used time series here instead of the usual set because
time series have an explicit order in their samples so the usual selection of
folds for cross validation won't apply. Instead, the cross validation will use
an approach like scikit-learn's
[`TimeSeriesSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html#sklearn.model_selection.TimeSeriesSplit).

The Cross Validation approach will be an optional configuraiton option, with
FormaK picking a sane default (TimeSeriesSplit).

### Metrics / Score Function

Metric to minimize: Normalized Innovation Squared (NIS). NIS is defined as

$$\tilde{z} = z - \hat{z}
S = H \overline{\Sigma} H^{T} + Q
NIS = \tilde{z}^{T} S^{-1} \tilde{z}$$

Roughly, it looks at the magnitude of the reading error $\tilde{z}$ vs the
expected variation $S$. FormaK uses the NIS metric because it doesn't require
ground truth data. An alternative, Normalized Estimation Error Squared (NEES)
looks at the errors vs ground truth. [3]

[3] NEES definition taken from "Kalman Filter Tuning with Bayesian
Optimization" by Chen et al.

One small problem with this measure:
1. The score will decrease consistently as the size of the error goes down (this is good, this means that the model is more accurate
2. The score will decrease consistently as the size of the variance goes up (this is bad, the optimization could increase the estimated variance to achieve a better score)

To remedy that, the metric to minimize is consistency instead of the true
minimum NIS. Taking advantage of the fact that we know that the errors should
follow a $\chi^{2}$ distribution, we can calculate consistency as follows:

$$\xi_{K_{r}} = NIS
\overline{\xi_K} = \sum_{r=1}^{r=N} \xi_{K_{r}}
F_{K}^{\chi^{2}} = Prob{\chi^{2}_n < \overline{\xi_K}
d_{i(k)} = F_{K}^{\chi^{2}} - \dfrac{i(k)}{N_{K}}
Consistency = \dfrac{1}{N_{K}} \sum_{1}^{N_{K}} \lvert d_{i_{(K)}} \rvert$$

[1] This approach to innovation filtering, referred to as editing in the text,
is adapted from "Advanced Kalman Filtering, Least Squares and Modeling" by
Bruce P. Gibbs (referred to as [1] or AKFLSM)
This consistency calculation is based on [2]
[Optimal Tuning Of A Kalman Filter Using Genetic Algorithms by Yaakov Oshman and Ilan Shaviv](https://arc.aiaa.org/doi/10.2514/6.2000-4558), but implemented with the NIS metric from [1].

Testing Notes:

- For testing, innovation, NIS and consistency, go to the math. Everything else in this feature is built on the metrics being evaluated correctly
- Test with mocked EKF so you can return a fixed state, covariance to get known innovation from readings

### State Management

All of the above design aspects have alluded to management tasks. This
funcitonality will be managed by a FormaK state machine for model creation
called `DesignManager`.

A state machine provides a straightforward concept for users interacting with
FormaK. Instead of having to re-invent a combination of tools each time a user
wants to use FormaK, they can instead follow one or more orchestrated flows
through the state machine to reach their goal.

The initial states for the state machine are:
- `Start`
- `Symbolic Model`
- `Fit Model`
- `Final Model`

An example state transition would be `Start` -> `Symbolic Model`. In code this
would look like:

```python
starting_state = formak.DesignManager(design_name='mercury')

symbolic_state = starting_state.symbolic_model(model=user_model)
```

The high level requirements for a hyper-parameter search can be slotted into
the various state transitions:
- The user will provide a symbolic model used to underly the estimator generation process in the `Start` -> `Symbolic Model` transition
- The user will provide the parameter space in the `Symbolic Model` -> `Fit Model` transition
- The user can provide configuration for the parameter search in the `Symbolic Model` -> `Fit Model` transition
- The user can provide configuration for the cross validation in the `Symbolic Model` -> `Fit Model` transition
- The score function will be hard coded for now within the FormaK model

Testing Notes:
- Every state transition should be covered. This can be tested by checking the return type. The desired functionality of the returned object should be independent of the state transition(s) used to get there.

#### Usability and the Discoverability Interface

To make the state machine easier to use, the public functions available on the
state machine will be minimized to those available for valid transitions with
three exceptions. The required arguments to each function represent the
required information to achieve the state transition.

The three exceptions:

First, a `history()` function can be called that will provide a user-readable
history of state transitions covered so far.

Second, a `transition_options()` function that will provide a list of function
names that can be called from the current state.

Third, a `search(desired_state)` function which will return a list of
transitions to take to reach the desired state. This search will be
accomplished dynamically by [inspecting the return
types](https://docs.python.org/3/library/inspect.html#introspecting-callables-with-the-signature-object)
of each transition out of the current state in a breadth first manner. From
there the transitions from each returned type can be used to identify
additional reachable states by taking two transitions. If this is not found the
process can be repeated to some known finite maximum depth. Breadth first
search should help identify a shortest path to the goal, although it may not
identify all paths.

Each state will be immutable.

Each state shall inherit from a common base class.

### New Class(es)

This feature may require implementing a new class or classes to interface with
scikit-learn tooling:
1. Metric function or class to pass to scikit-learn (see the [`make_scorer`](https://scikit-learn.org/stable/modules/model_evaluation.html#defining-your-scoring-strategy-from-metric-functions) interface)
2. Dataset class. Most scikit-learn tooling deals in vectors and that may not be fully suitable for sensor data

## Experiments

- State Discovery via [inspection](https://docs.python.org/3/library/inspect.html#introspecting-callables-with-the-signature-object)
- Compare grid search and randomized search for parameter evaluation. Optionally, compare with [other search algorithms](https://scikit-learn.org/stable/modules/classes.html#hyper-parameter-optimizers).

## Feature Tests

Compare a correct model with a dataset with arbitrary, incorrect +- 5 std dev
noise inserted around other noise that is normally distributed. Selecting
parameter should be less than 5. This can be repeated for multiple correct
values of innovation filtering, say in the range 4-10.

By using synthetic data, a known correct answer can be used and the complexity
of the feature test can be focused on demonstrating the interface (instead of
say demonstrating the construction of a complex model to deal with complex
real-world phenomena).

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
