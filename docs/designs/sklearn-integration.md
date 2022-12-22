# Scikit-Learn integration

Author: Buck Baskin @bebaskin
Created: 2022-09-26
Updated: 2022-12-21
Parent Design: [designs/formak_v0.md](../designs/formak_v0.md)
See Also: [designs/python_library_for_model_evaluation.md](../designs/python_library_for_model_evaluation.md)
Status: 7. Unit Testing


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

This design provides the initial implementation of third of the Five Keys
"Integration to scikit-learn to leverage the model selection and parameter
tuning functions". Scikit-learn is a common library who's interface is
replicated many places (e.g. dask-ml for scaling up machine learning tasks)
that's a good place to start with for an easy to use library.

Why is scikit-learn and machine learning relevant? Conceptually, a detailed,
physical model derived from first principles describes both one complex model,
as well as a space of models derived via simplifications, enhancements or even
disconnected approximations from the original model. Using data from the system
we hope to describe, we can select the appropriate model from the space. This
process is very analogous to a machine learning model, where we have one idea of
how to approximate the system and want to select machine learning models (in a
more algorithmic sense of the term models) and their parameters to best fit
data.

### The Dream

In the end, my hope is that the user can provide an arbitrarily complex
description of the system as a model and provide data and auto-magically get a
best fit approximation to their system. Providing a more complicated model
provides more of a space for discovering improvements to the final system in the
same way providing more data can improve the final system. The "auto-magic"
doesn't come from anything magical; instead, it comes from accumulating
knowledge and how to use it in one place where the final level (improved
knowledge) can also improve the final system above and beyond that which could
be achieved by the user alone.

First, I'll do an overview of scikit-learn and its key elements (as it relates
to FormaK). Second, I'll discuss the solution approach and the tooling to
implement the approach. To conclude, I'll describe the feature tests that will
help track our progress and the roadmap for landing this design.

## Context

To start with, I'm going to take a moment to introduce key elements from
scikit-learn. Integrating, reusing or replicating these elements will form the
key motivation for this design. If you're more interested in how that
materializes into the design, skip ahead to the Solution Approach section.

### Scikit-Learn

Taking a look at the [scikit-learn home page](https://scikit-learn.org/stable/),
a few key elements stand out to me as likely sources of inspiration and
integration:
- [Regression](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning)
- [Dimensionality Reduction](https://scikit-learn.org/stable/modules/decomposition.html#decompositions) (aka Decomposing Signals)
- [Model Selection](https://scikit-learn.org/stable/model_selection.html#model-selection)
- [Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing)

In addition to that, taking a look at some of the bigger areas of functionality
in the user guide:
- [Unsupervised Learning](https://scikit-learn.org/stable/unsupervised_learning.html)
- [Inspection](https://scikit-learn.org/stable/inspection.html) (aka Interpretability)
- [Visualizations](https://scikit-learn.org/stable/visualizations.html)
- [Dataset Transformations](https://scikit-learn.org/stable/data_transforms.html)
- [Pipelines](https://scikit-learn.org/stable/modules/compose.html#pipeline) and [Feature Unions](https://scikit-learn.org/stable/modules/compose.html#feature-union)
- [Dataset tools (loading, generating, etc)](https://scikit-learn.org/stable/datasets.html)
- [Performance](https://scikit-learn.org/stable/computing.html)
- [Model Persistence](https://scikit-learn.org/stable/model_persistence.html)
- [Best Practices](https://scikit-learn.org/stable/common_pitfalls.html)

#### Regression 

From scikit-learn:
"[Regression](https://scikit-learn.org/stable/supervised_learning.html#supervised-learning):
Predicting a continuous-valued attribute associated with an object."

Regression and its use as a predictor of a continuous valued attribute are the
essential function for the models we hope to evaluate by using historical data
to predict new values from some prior inputs (model state, control inputs).

#### Dimensionality Reduction

From scikit-learn: "[Dimensionality
reduction](https://scikit-learn.org/stable/modules/decomposition.html#decompositions):
Reducing the number of random variables to consider."

Dimensionality reduction is a common practice in machine learning for selecting
a subset of elements. This is helpful for both improving machine learning model
robustness (simpler models are easier to understand, easier to characterize and
less prone to overfitting) and improving model compute performance (a simpler
model should be faster to compute and uses less memory).

Dimensionality reduction is useful for FormaK as a tool for improving model
robustness (simpler models are easier to understand, easier to characterize and
less prone to overfitting) as well as improving model compute performance (a
simpler model should be faster to compute and uses less memory).

Here, the analogy is quite literally, although the techniques may vary more than
the analogy would suggest. Some physically useful approximations (e.g. the small
angle approximation that sin(theta) ~= theta) don't have generic algorithmic
analogies (or maybe they do: functions can be approximated by their first or Nth
order Taylor series, which is close family to the linearization used in the
Extended Kalman Filter). See also, the Notes on Taylor Series in the Appendix.

#### Model Selection

From scikit-learn: "[Model
selection](https://scikit-learn.org/stable/model_selection.html#model-selection):
Comparing, validating and choosing parameters and models."

As we lean into the single complex model, model selection as a general idea is a
key feature; however, it also extends to relevant additional considerations
beyond what you might guess from the name. Model selection also encompasses
developing metrics, validation across sets of test data, selecting model
parameters and visual tools for intuitive understanding of model performance.

#### Preprocessing

From scikit-learn:
"[Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing):
Feature extraction and normalization."

The standard form for scikit-learn preprocessing is to center the data (subtract
the mean) and make it approximately unit variance (divide by the standard
deviation). This is also a good form for a Kalman filter and a baseline for
understanding if a model is likely a good fit (say noise is small and normally
distributed so likely from random unpredictable processes) or biased in some way
(an indication that the model is insufficient for describing the data well.
- There are also non-linear transformations we can leverage such as mapping to a uniform or Gaussian distribution. 
- The preprocessing library has support for generating new [polynomial features](https://scikit-learn.org/stable/modules/preprocessing.html#generating-polynomial-features) from existing features to augment the state. This seems exciting as a way to approximate expected but unknown physical phenomena. If we have a position/velocity model, can we generate expected accelerations or jerk behavior or something like that (e.g. velocity ~= jerk * t^2 + accel * t) and if we have generic polynomial terms we don't need to know the coefficients ahead of time.

#### Unsupervised Learning

[Unsupervised learning](https://scikit-learn.org/stable/unsupervised_learning.html)
covers a lot of topics and the task that we're hoping to achieve is
unsupervised. We don't have known "right" models, we have detailed models of
some systems and can compare them with simpler models to guide some aspects of
the project, but models could always be more complicated or the complicated
model might not be the optimal fit for the constraints. Therefore, the
techniques to recover useful insight (learning) from data alone are directly
useful for FormaK.

Some of the most promising areas in supervised learning I'm interested to
explore:

##### [Gaussian mixture models](https://scikit-learn.org/stable/modules/mixture.html)

Approximating a process as a mix of a finite set of Gaussian distributions.
This seems most likely to be if there are multiple confusing sources of noise
in a single area (e.g. multiple processes could have the same effect on a
sensor reading) and teasing those apart or identifying when two distributions
are indistinguishable and we could simplify the model by modeling just the one
distribution.

##### [Manifold learning](https://scikit-learn.org/stable/modules/manifold.html)

"Manifold Learning can be thought of as an attempt to generalize linear
frameworks like Principle Components Analysis (PCA) to be sensitive to
non-linear structure in data. Though supervised variants exist, the typical
manifold learning problem is unsupervised: it learns the high-dimensional
structure of the data from the data itself, without the use of predetermined
classifications." This seems of particular use for identifying simplification
functions over the model, although I don't know if it'd be compatible with
analyzing a symbolic model or if it'd have to round-trip through some data
generation first. Perhaps the data generation is a good abstraction, if the
model only ever generates data that looks like a low dimensional space, we can
jump to that low dimensional space.

##### [Matrix factorization/PCA](https://scikit-learn.org/stable/modules/decomposition.html)

PCA aims to find a smaller subset of vectors that can explain the variability
of the full set. This is at least directly relevant to model simplification
(e.g. identifying if two states can be represented as one without losing
information, or otherwise quantifying that we'd say be able to represent 90% of
the data with the simplification). This approach is probably worth trying
first, but it may be superceded by manifold learning because models are almost
certainly non-linear
    - [Incremental PCA](https://scikit-learn.org/stable/modules/decomposition.html#incremental-pca)
may be of special use for the timeseries data

##### [Biclustering](https://scikit-learn.org/stable/modules/biclustering.html)

I think this is the algorithm that solves the state layout idea that I'd had in
mind. Biclustering tries to find block diagonals that are connected and leave
the rest as zeros/nearly zero/sparse. This would allow for densely packing
subsets of the state-vector that depend on the densely packed bit (and would be
more likely to be shared in cache) and partially ordering the rest that aren't
connected (by their own connections). 

##### [Covariance estimation](https://scikit-learn.org/stable/modules/covariance.html)

This may align with a Kalman filter-based approach I'm considering, or
supersede it if the [robust covariance
estimation](https://scikit-learn.org/stable/modules/covariance.html#robust-covariance-estimation)
is what I hope it is based on the name. There's even an example of ["Separating
inliers from outliers using a Mahalanobis
distance"](https://scikit-learn.org/stable/auto_examples/covariance/plot_mahalanobis_distances.html).

##### [Outlier Detection](https://scikit-learn.org/stable/modules/outlier_detection.html)

Online outlier detection is useful when using the model after the fact, but it
may also be useful to do offline bulk outlier detection as a way to identify
trends that the model is insufficient for describing.

##### Other Unsupervised Learning

Some of the other elements within unsupervised learning are certainly
interesting, but it's less clear to me how they'll fit into the FormaK design:

- [Clustering](https://scikit-learn.org/stable/modules/clustering.html): I think this will take some more abstraction vs the typical scikit-learn examples to figure out how clustering will apply to the modeling problem. This [article on time-series clustering](https://towardsdatascience.com/time-series-clustering-deriving-trends-and-archetypes-from-sequential-data-bb87783312b4) offers some insights into the types of techniques that could be applied to time series clustering and what they're hoping to solve (and some may fall outside the scope of scikit-learn tooling). The focus is on identifying trends a human might see in data in an automated fashion and we could apply this trend-identification as way to quantify cases where the data's expected trend from the model deviates.
- [Density Estimation](https://scikit-learn.org/stable/modules/density.html): "One other useful application of kernel density estimation is to learn a non-parametric generative model of a dataset in order to efficiently draw new samples from this generative model." I think this opens more questions than it answers. One area that I'd be curious to explore is: Can you use the generative model based on the symbolic model and the generative model learned from data to compare the two and quantify the effectiveness of the symbolic model?

#### Inspection (aka Interpretability)

From scikit-learn: "The [sklearn.inspection](https://scikit-learn.org/stable/modules/classes.html#module-sklearn.inspection)
 module provides tools to help understand the predictions from a model and what
affects them. This can be used to evaluate assumptions and biases of a model,
design a better model, or to diagnose issues with model performance."

#### Visualizations

From scikit-learn: "[A] simple API for creating visualizations for machine
learning"

#### Dataset Transformations

From scikit-learn: "scikit-learn provides a library of transformers, which may
clean (see [Preprocessing
data](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing)),
reduce (see [Unsupervised dimensionality
reduction](https://scikit-learn.org/stable/modules/unsupervised_reduction.html#data-reduction)),
expand (see [Kernel
Approximation](https://scikit-learn.org/stable/modules/kernel_approximation.html#kernel-approximation))
or generate (see [Feature
extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#feature-extraction))
feature representations."

#### Pipelines and Feature Unions

[Pipelines](https://scikit-learn.org/stable/modules/compose.html#pipeline) form
an ordered sequence of tools. [Feature
Unions](https://scikit-learn.org/stable/modules/compose.html#feature-union)
allow for combining multiples outputs into the input of a single element.
Scikit-learn uses the term transformer to refer to the generic step in the
pipeline.

#### Dataset tools (loading, generating, etc)

Three parts:
- [toy datasets](https://scikit-learn.org/stable/datasets/toy_dataset.html)
- [help downloading real world datasets](https://scikit-learn.org/stable/datasets/real_world.html)
- [generating datasets](https://scikit-learn.org/stable/datasets/sample_generators.html)

#### Computing (aka Performance)

Three parts:
- [Strategies to scale computationally [for] bigger data](https://scikit-learn.org/stable/computing/scaling_strategies.html)
- [Computational Performance](https://scikit-learn.org/stable/computing/computational_performance.html)
- [Parallelism, resource management, and configuration](https://scikit-learn.org/stable/computing/parallelism.html)

#### [Model Persistence](https://scikit-learn.org/stable/model_persistence.html)

"After training a scikit-learn model, it is desirable to have a way to persist
the model for future use without having to retrain"

#### [Best Practices](https://scikit-learn.org/stable/common_pitfalls.html)

"Illustrate some common pitfalls and anti-patterns that occur when using
scikit-learn"

## Solution Approach

The basic step will be to integrate the ui.Model into an interface like a
scikit-learn model. This will allow for easy integration with scikit-learn
pipelines, model selection and other functionality.

The key classes involved are:
- `py.Model`: (new) Class encapsulating the model for running a model efficiently in Python code
- A scikit-learn interface, starting with`.fit`and `.predict`for regression

The ui library will likely also need some updates to support the configuration
of the scikit-learn like behavior.

Keeping the machine learning analogy in mind, the key elements from
scikit-learn for this design is regression. This will keep the design focused
and unlock future advances.

### The Road Ahead

I mentioned it above in passing above, but I want to repeat it again for
emphasis:

Conceptually, a detailed, physical model derived from first principles describes
both one complex model, as well as a space of models derived via
simplifications, enhancements or even disconnected approximations from the
original model. Using data from the system we hope to describe, we can select
the appropriate model from the space. This process is very analogous to a
machine learning model, where we have one idea of how to approximate the system
and want to select machine learning models (in a more algorithmic sense of the
term models) and their parameters to best fit data.

As a psuedo-roadmap for future integrations, model selection, dimensionality
reduction and the preprocessing steps to augment a model with extra features
are promising for areas where a scikit-learn integration (being able to treat
the model as yet another scikit-learn regression) could easily yield a lot of
fruit. This would allow for features like:
1. Given two models: a complicated model with position/velocity/acceleration/heading and a simple model with position/velocity, match fake data with position, velocity and select to the simpler model (Model selection)
2. Given a complicated model with position/velocity/acceleration/heading, match fake data with position, velocity and down select to a simpler position/velocity model (Preprocessing/Dimensionality Reduction)
3. Given a simple model with position/velocity, match fake data with position, velocity and acceleration and augment to the more complicated model (Preprocessing/Model Augmentation)

## Feature Tests

This feature is specific to the Python interface. There will be X feature tests:
1. Set up a simple scikit-learn pipeline and run it against the code (basic UI)
2. Generate some fake data for a known model and fit a model to it (Regression)

## Road Map and Process

1. Write a design
2. Write a feature test(s)
3. Build a simple prototype
4. Pass feature tests
5. Refactor/cleanup
6. Build an instructive prototype (e.g. something that looks like the project vision but doesn’t need to be the full thing)
7. Add unit testing, etc
8. Refactor/cleanup
9. Document the functionality
10. Write up successes, retro of what changed (so I can check for this in future designs)
	1. Good
	2. Bad
	3. What was added?
	4. What was removed from the design?
	5. Anything else?

## Post Review

### 2022-12-21

#### Retro: The Good

- I think the overall idea and borrowing UI design is a helpful place to start
- Self review provided some good ideas

#### Retro: The Bad

- The delays
- Mixing in linting with something that was already going slowly

### 2022-12-19

Some thoughts:

It was tough to get myself back into completing the PR (by dates it took more
than 2 months of mostly not much activity.

That said, fixing the PR just took about 30 min to an hour of debugging and
re-reading through stuff and being frustrated with past self for mixing things
up. I'm hoping that some ammount of typing in the future will make it easier to
sort out these kinds of mistakes with a tool instead of requiring me to look
through to see where I'd mixed things up myself.

Self reviewing code was a win: I came out with some ideas for improving the
code that I'd missed when I was heads down writing the code. I'll need to have
a place to keep track of code review comments I've made but didn't implement.

Adding all the linting tools as part of this PR will make further review more
burdensome because it's touching lots of files. Probably a net win for
improving the codebase, but I still haven't gotten to mypy yet.

### 2022-10-06

#### Interface

One of the things that was poorly specified in the original design was what
part of the scikit-learn interface that I'd adopt. In the end, the list is:

The common stuff:

- `fit`: common to all scikit-learn use cases
- `score`: common
- `get_param`: common, useful for the in-place modification scikit-learn does to estimators
- `set_param`: common, useful for the in-place modification scikit-learn does to estimators

Inspired by the Covariance models:

- `mahalanobis`: from the Covariance class, pretty easily falls out of the Extended Kalman Filter design

Inspired by the manifold learning models:

- `transform`: This takes the view that the Kalman Filter can also be used as a method to transform an input sequence into a series of errors/innovations based on its model.
- `fit_transform`: Same as transform, except that because both transform and fit calculate innovations we can save some computation by performing them together


#### Model location

The location for integrating with scikt-learn moved from the `ui.Model` to the
`python.EKF` class.

I'd originally written the scikit-learn interface as part of the `ui.Model`,
but I essentially had to re-implement the `python.EKF` class in order to
implement fit and score. This also created some circular dependencies because
the `formak.python` library loads `formak.ui` but `ui.Model` would want to load
`python.EKF` to perform the optimization.

This change solidifies the `ui` classes as helpers for the symbolic model and
then `python` as one of the ways to fit or run the model.

#### Documentation

Added documentation as a step in the design / PR process. 

Also, adds having a diff in the docs folder (not including the designs) as a
Github Action. Not required, but will surface if there have been no docs
changes.

## Appendix

### Citing scikit-learn

If you use scikit-learn in a scientific publication, we would appreciate citations to the following paper:
Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.
Bibtex entry:
```
@article{scikit-learn,
 title={Scikit-learn: Machine Learning in {P}ython},
 author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
         and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
         and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
         Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
 journal={Journal of Machine Learning Research},
 volume={12},
 pages={2825--2830},
 year={2011}
}

```

If you want to cite scikit-learn for its API or design, you may also want to consider the following paper:
API design for machine learning software: experiences from the scikit-learn project, Buitinck et al., 2013.
Bibtex entry:
```
@inproceedings{sklearn_api,
  author    = {Lars Buitinck and Gilles Louppe and Mathieu Blondel and
               Fabian Pedregosa and Andreas Mueller and Olivier Grisel and
               Vlad Niculae and Peter Prettenhofer and Alexandre Gramfort
               and Jaques Grobler and Robert Layton and Jake VanderPlas and
               Arnaud Joly and Brian Holt and Ga{\"{e}}l Varoquaux},
  title     = {{API} design for machine learning software: experiences from the scikit-learn
               project},
  booktitle = {ECML PKDD Workshop: Languages for Data Mining and Machine Learning},
  year      = {2013},
  pages = {108--122},
}

```
