# Innovation Filtering

Author: Buck Baskin @bebaskin
Created: 2023-08-04
Updated: 2023-08-04
Parent Design: [designs/sklearn-integration.md](../designs/sklearn-integration.md)
See Also: [designs/python_library_for_model_evaluation.md](../designs/python_library_for_model_evaluation.md), [designs/cpp_library_for_model_evaluation.md](../designs/cpp_library_for_model_evaluation.md)
Status: 1. Design


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
selection and parameter tuning functions". The end result will be updates to
Python and C++ models, but the primary effort will be improving the
scikit-learn tooling. The current implementation provides some of the math for
the EKF (process update, sensor update); however, that leaves the models open
to spurious readings. In the current implementation, if a FormaK filter gets a
reading that is almost infinite, say 1e300, the response of the filter will be
proportional and the response will cause the EKF to diverge as one or more of
the states jump to positive infinity. This creates a gap in the filter that
violates one of the FormaK values: easy to use (perhaps in this case it's
helpful to think of this as easy to use *correctly*).

This problem is known as jump detection. As the filter considers a sensor
update the filter measures the divergence of the reading from what the model
expects and then can reject readings with a divergence that is too large to be
acceptable.

How should the filter measure this divergence? The filter can use the
measurement innovation $$\tilde{y} = y - h(x)$$ where y is the measurement, h
is the measurement model and x is the current state estimate and $\tilde{y}$ is
the resulting innovation. Given the innovation, the definition of too large can
be calculated from the expected covariance: $$\tilde{y}^{T} C^{-1} \tilde{y} -
m > k \sqrt{2m}$$ where $C$ is the expected covariance of the measurement, $m$
is the dimension of the measurement vector and $k$ is the editing threshold. If
this inequality is true, then the measurement is "too large" and should be
filtered (edited). [1]

[1] This approach to innovation filtering, referred to as editing in the text,
is adapted from "Advanced Kalman Filtering, Least Squares and Modeling" by
Bruce P. Gibbs (referred to as [1] or AKFLSM)

Innovations take advantage of the property that errors are white (normally
distributed) when all models are correct and, when operating in steady state,
most variation in the sensor reading is expected to come from the noise in the
sensor instead of noise in the state. Intuitively, if the sensor is
consistently providing low variance updates, the state should converge further
based on that information until the noise in the state relative to the sensor
is low.

By implementing innovation filtering, the filter will become much more robust
to errors in the sensor, the model and the overall system. By hooking into
values that are already calculated as part of the sensor update there should be
minimal additional performance cost.

## Solution Approach

The primary class of interest will be the Extended Kalman Filter (EKF)
implementation and revising the sensor update specifically. This change won't
require a large change to the structure of the EKF generation in Python or C++,
but it will be fairly math heavy to ensure the changes are working as intended.

### Innovation Filtering

#### Math

The math has been outlined above:
$$\tilde{y} = y - Hx$$
$$C = HPH^{T} + R$$
$$\tilde{y}^{T} C^{-1} \tilde{y} - m > k \sqrt{2m}$$

To provide the maximum value from the innovation filtering, the calculation of
whether or not to reject the sensor update should be made as early as possible
(as soon as $\tilde{y}$ is calculated). That way there is no unnecessary wasted
computation.

#### Filter Health

Innovation filtering makes the assumption that the model is correct and the
sensor readings may be errant; however, repeated filtering may indicate that
the filter has diverged. In this initial implementation, the count of
innovations that are filtered will be tracked online per-measurement update.


#### Other Implementation Details

- Use if constexpr for the innovation filtering calculation if turned on ($k > 0$)
- Use a continuous value for the editing threshold $k$. Provide $k = 5$ as a default edit threshold based on the textbook recommendation [1]

### Model Selection

In AKFLSM there isn't an immediate discussion of how to select the editing
threshold $k$ based on time series. The design will provide a sane default, but
it will also be useful to provide a data-driven function to select the value
based on prior data.

The editing threshold will be selected via a hyperparameter selection process.
The high level process is composed of the following elements (from the
[scikit-learn user guide](https://scikit-learn.org/stable/modules/grid_search.html)):
- The estimator (with fit and predict operations)
- a parameter space
- a method for searching or sampling candidates
- a cross-validation scheme
- a score function

The parameter space for this problem is positive floats. The problem could be
thought of as positive integers (how many standard deviations are acceptable)
but the positive floats should be easier to optimize as a continuous parameter.

#### Parameter Search

The default method for parameter optimization is
[exhaustive grid search](https://scikit-learn.org/stable/modules/grid_search.html#exhaustive-grid-search).
For this problem of selecting a single or reduced set of parameters it should
be okay to use the baseline; however, the design will include an experiment to
compare between grid search and randomized search. The benefit of randomized
search is that a specific budget can be set and the approach will, with high
probability, find the best result in the same search space as the grid search
but with fewer iterations.

#### Cross Validation

The
[cross-validation approach](https://scikit-learn.org/stable/modules/cross_validation.html)
suggested by scikit-learn to choose the hyperparameter with the highest
cross-validation score:
1. Hold aside a test time series that does not overlap with the rest of the data set
2. Within the remaining data, the training time series, subdivide into different folds where one part of each fold is used to train the model and a subset is used for validation. By selecting many folds, we can reduce the chance that the parameters are overfit to noise in a particular test evaluation.
3. Perform final scoring against the held out test data.

One thing to note: I've used time series here instead of the usual set because
time series have an explicit order in their samples so the usual selection of
folds for cross validation won't apply. Instead, the cross validation will use
an approach like scikit-learn's
[`TimeSeriesSplit`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html#sklearn.model_selection.TimeSeriesSplit).

#### Metric / Score Function

Metric to minimize: Normalize innovation squared (NIS). NIS is defined as

$$e_z = z - h(x)
S = H P H^{T} + R
NIS = e_z^{T} S^{-1} e_z$$

Roughly, it looks at the magnitude of the reading error $e_z$ vs the expected
variation $S$. FormaK uses the NIS metric because it doesn't require ground
truth data. An alternative, "NEES" looks at the errors vs ground truth.

One small problem with this measure:
1. The score will decrease consistently as the size of the error goes down (this is good, this means that the model is more accurate
2. The score will decrease consistently as the size of the variance goes up (this is bad, the optimization could increase the estimated variance to achieve a better score)

To remedy that, the metric to minimize is consistency instead of the true
minimum score. Taking advantage of the fact that we know that the errors should
follow a $\chi^{2}$ distribution, we can calculate consistency as follows:

$$ \xi_{K_{r}} = NIS
\overline{\xi_K} = \sum_{r=1}^{r=N} \xi_{K_{r}}
F_{K}^{\chi^{2}} = Prob{\chi^{2}_n < \overline{\xi_K}
d_{i(k)} = F_{K}^{\chi^{2}} - \dfrac{i(k)}{N_{K}}
Consistency = \dfrac{1}{N_{K}} \sum_{1}^{N_{K}} \lvert d_{i_{(K)}} \rvert$$

### Visualization

Optional extension: Visualizations
1. The magnitude of normalized innovations for different parameter tunings over a sequence of reading. This should help the user understand if the parameter is selecting models that are converging and if there are portions of the dataset where many models aren't well fit.
2. The chi2 distribution for innovations for different parameter tunings. I imagine this would look a lot like ROC curves

### Testing for Better Metrics

- Testing, innovation, consistency, go to the math. Everything else in this feature is built on the metrics being evaluated correctly
- Test with mocked EKF so you can return a fixed state, covariance to get known innovation from readings

### New Class(es)

This feature may require implementing a new class or classes to interface with
scikit-learn tooling:
1. Metric function or class to pass to scikit-learn (see the [`make_scorer`](https://scikit-learn.org/stable/modules/model_evaluation.html#defining-your-scoring-strategy-from-metric-functions) interface)
2. Dataset class. Most scikit-learn tooling deals in vectors and that may not be fully suitable for the

## Experiments

- Compare grid search and randomized search for parameter evaluation. Optionally, compare with [other search algorithms](https://scikit-learn.org/stable/modules/classes.html#hyper-parameter-optimizers).

## Feature Tests

### Innovation Filtering

- x, y, heading, velocity model
- Provide heading readings, expect rejecting 180 degree heading errors
	- Nonlinear model provides clear divergence signal

Additional testing should be performed on measurements and filters of multiple sizes

### Model Selection

1. Compare a correct model with a dataset with arbitrary, incorrect +- 5 std dev noise inserted around other noise that is normally distributed. Selecting parameter should be less than 5

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
