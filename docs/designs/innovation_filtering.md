# Innovation Filtering

:Author: Buck Baskin @buck@fosstodon.org
:Created: 2023-08-04
:Updated: 2023-09-19
:Parent Design: [designs/sklearn-integration.md](../designs/sklearn-integration.md)
:See Also: [designs/python_library_for_model_evaluation.md](../designs/python_library_for_model_evaluation.md), [designs/cpp_library_for_model_evaluation.md](../designs/cpp_library_for_model_evaluation.md)


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
measurement innovation $$\tilde{z} = z_{t} - \hat{z_{t}}$$ where $z_{t}$ is the
measurement, $\hat{z_{t}}$ the predicted reading and $\tilde{z}$ is the
resulting innovation. Given the innovation, the definition of too large can be
calculated from the expected covariance: $$\tilde{z}^{T} S^{-1} \tilde{z} - m >
k \sqrt{2m}$$ where $S$ is the expected covariance of the measurement, $m$ is
the dimension of the measurement vector and $k$ is the editing threshold. If
this inequality is true, then the measurement is "too large" and should be
filtered (edited). [1] [2]

[1] This approach to innovation filtering, referred to as editing in the text,
is adapted from "Advanced Kalman Filtering, Least Squares and Modeling" by
Bruce P. Gibbs (referred to as [1] or AKFLSM)
[2] The notion follows the conventioned defined in the
[Mathematical Glossary](../mathematical-glossary.md) which itself is based on
"Probabilistic Robotics" by Thrun et al.

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

The math has been outlined above [1]:
$$\tilde{z} = z - \hat{z}$$
$$S = H \overline{\Sigma} H^{T} + Q$$
$$\tilde{z}^{T} S^{-1} \tilde{z} - m > k \sqrt{2m}$$

To provide the maximum value from the innovation filtering, the calculation of
whether or not to reject the sensor update should be made as early as possible
(as soon as $\tilde{z}$ is calculated). That way there is no unnecessary wasted
computation.

Revision:
The above notation:
$$\tilde{z}^{T} S^{-1} \tilde{z} - m > k \sqrt{2m}$$

can be reorganized as

$$\tilde{z}^{T} S^{-1} \tilde{z} > k \sqrt{2m} + m$$

Which has the benefit that the left side is reading-dependent and the right
side is a constant expression for each reading.

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
minimum score. Taking advantage of the fact that we know that the errors should
follow a $\chi^{2}$ distribution, we can calculate consistency as follows:

$$ \xi_{K_{r}} = NIS
\overline{\xi_K} = \sum_{r=1}^{r=N} \xi_{K_{r}}
F_{K}^{\chi^{2}} = Prob{\chi^{2}_n < \overline{\xi_K}
d_{i(k)} = F_{K}^{\chi^{2}} - \dfrac{i(k)}{N_{K}}
Consistency = \dfrac{1}{N_{K}} \sum_{1}^{N_{K}} \lvert d_{i_{(K)}} \rvert$$

This consistency calculation is based on [2]
[Optimal Tuning Of A Kalman Filter Using Genetic Algorithms by Yaakov Oshman and Ilan Shaviv](https://arc.aiaa.org/doi/10.2514/6.2000-4558), but implemented with the NIS metric from [1].

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

### 2023-08-13

Revise to a consistent mathematical notation. Also document a revision in how
the math for innovation filtering is stated.

### 2023-09-18

[Someone Dead Ruined My Life... Again.](https://www.youtube.com/watch?v=qEV9qoup2mQ)

A refactor ruined my ~life~ implementation of this feature.

In the previous implementation
[PR #17](https://github.com/buckbaskin/formak/pull/17/), I'd run into an issue
where the state elements and reading elements were not in the order I expected
or specified. For a state `{"mass", "z", "v", "a"}` the state would be
generated as `["a", "mass", "v", "z"]`, so `np.array([[0.0, 1.0, 0.0,0.0]])`
sets the `mass` to `1.0`, but if I renamed `z` to `b` it'd be silently setting
`b` to 1.0 instead.

This potential for reordering will ultimately be a feature (where the code
generation can reorder values to better pack into memory) but for now it was
only a surprise and it led to a failing test (better to fail the test than to
silently fail though).

When I went to implement innovation filtering, I again ran into this issue.
Logically, the only solution would be to refactor how all data types are stored
and passed around in the Python implementation of FormaK.

And so, I walked deep into the forest of refactoring. To cut a long story
short, this took way longer than I wanted. The first notes I have for the start
of the refactor is 2023-08-13 and the end of the refactor dates roughly to
2023-09-07, so I spent nearly 3.5 weeks on this refactor to "avoid confusion"
and instead introduced lots of internal confusion as I was refactoring and
instead introduced much consternation.

Today marks the end of the design, partly because I have some tests passing but
mostly because I want to call it done enough, move to the next thing and
revisit it (and test it more) if I find that it isn't working as intended.

Beware the forest of refactoring.

## 2023-09-19

The second aspect of this design around model selection was dropped in favor of
completing the innovation filtering portion of the design. Model selection
integration will be revisited in a future design.
