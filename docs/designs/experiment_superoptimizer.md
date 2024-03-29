# Experiment: Superoptimization

:Author: Buck Baskin [@buck@fosstodon.org](https://fosstodon.org/@buck)
:Created: 2024-02-14
:Updated: 2024-02-27
:Parent Design: [designs/python_library_for_model_evaluation.md](../designs/python_library_for_model_evaluation.md)

## Overview

FormaK aims to combine symbolic modeling for fast,
efficient system modelling with code generation to create performant
code that is easy to use.

The Five Key Elements the library provides to achieve this user experience are:

1. Python Interface to define models
2. Python implementation of the model and supporting tooling
3. Integration to scikit-learn to leverage the model selection and parameter tuning functions
4. C++ and Python to C++ interoperability for performance
5. C++ interfaces to support a variety of model uses

This design focuses on experimenting with the possible performance benefits
from the straightforward (common subexpression elimination) to the magical:
super-optimization.

The performance is relevant in two key ways:

1. Evaluating the runtime of the output program vs the current system
2. Evaluating the compile time of the super-optimizing program to see if it is sufficiently fast to be usable

This design is experimental in nature, so the end goal is only to collect this
data to establish a data point from which future work can proceed. There is no
goal to have this design be a full feature of the project.

### Superoptimization

[Superoptimization](https://en.wikipedia.org/wiki/Superoptimization) is the
process of searching to find the optimal code sequence to compute a function.

For this design, the approach will be to perform a search on the compute graph
to find the sequence of operations that lead to the fastest possible
computation. To do that search, a CPU model will be used to allow for mapping
operations to a more detailed sense of time (vs assigning each operation a
fixed time), primarily focusing on modeling memory latency and CPU instruction
pipelining. This will allow the search to model the state of the CPU at each
instruction and have a better approximation of the total time to compute the
sequence.

## Solution Approach

### Search

By taking a graph-based approach, the search algorithm `A*` (A-star) can be
used to speed up the search with heuristics. The key to using `A*` search
effectively is a heuristic that is quick to compute, admissible and consistent.

[Admissible](https://en.wikipedia.org/wiki/Admissible_heuristic)

> a heuristic function is said to be **admissible** if it never overestimates
> the cost of reaching the goal, i.e. the cost it estimates to reach the goal
> is not higher than the lowest possible cost from the current point in the
> path

[Consistent](https://en.wikipedia.org/wiki/Consistent_heuristic)

> a heuristic function is said to be **consistent**, …  if its estimate is
> always less than or equal to the estimated distance from any neighboring
> vertex to the goal, plus the cost of reaching that neighbor.

The quick to compute part is relevant because the end to end search time could
end up being slower if it’s faster to evaluate some large portion of the graph
than to evaluate the heuristic function. In this case, given that the CPU model
may grow to be somewhat complex, the heuristic should have a low hurdle to step
over (or a high ceiling to step under?).

### CPU Model

The CPU model used in this superoptimization will focus on a few key features
of CPUs: pipelining of independent operations and memory load latency. This
focus comes because the modeling of these two effects is approximately
tractable and the two effects should have a straightforward implications for
the output graph:

- If you can change the order of two compute operations so more are running in parallel via pipelining than the overall compute will be faster.
- If you can load some memory earlier, than later computations may not need to wait as long

For kicks, they’re also parts of the CPU I’m interested in modeling.

## Feature Tests

The feature test for this will be setting up a simple compute graph and running
the superoptimization experiment on the graph. Given a simple-enough graph, it
should be feasible to predetermine the optimal result and match it to the
algorithm's result.

From there, the time to run the super-optimized version will be compared to the
time to run the basic version (with common subexpression elimination) and the
time to run the superoptimization can be compared to the time to run the common
subexpression elimination on its own.

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

### 2024-02-27

Overall, I'd say the experiment is a success. I added an `A*` implementation
and did some code search with a CPU model.

The key risks to keep an eye out for:
1. Fully understanding the possible operations in use by `sympy`, and how that maps (almost but not quite 1:1) to the desired operations set
2. Maximum runtime: the current version can go from fast for simple operations (<< 1 second) to unusably slow when it times out (10s of seconds)

I'd be curious to do some more literature review to understand potential areas
for improvements.

In addition, a good heuristic seems essential to reduce states evaluated and
overall performance would be important for the end user experience. Perhaps
this is a flag that can be turned on for release builds (evaluating a slightly
suboptimal model should probably be fine).

#### Wins

I'm really happy with the architectural approach/design of the `A*`
implementation. I think it makes intuitive sense and has a nice functional
programming aesthetic. That said, I've been plagued by nagging doubts that it
may not be correct. Given the time-bounded experimental nature I didn't spend
more time testing, but would like to in the future (e.g. make assertions on
evaluating the minimum number of states for a small-to-medium complexity
problem).

#### Opportunities for Improvement

##### Multi-scale search

If I have a set of expressions and the topological mapping between them, I can
only evaluate the frontier of potential elements to evaluate for expressions
that don't have topological dependencies (given the currently available set).
This will play nicely with common subexpression elimination which generates the
nice tree of values.

Multi-scale search should also help with register spilling in that it'll put
more constraints on the values to be evaluated, which should reduce the
potential number of values "in-flight". With the topological mapping, a set of
up to `number of registers` (maybe `number of registers / 2` for binary
operations) functions can be evaluated in parallel via the
[Coffman–Graham algorithm](https://en.wikipedia.org/wiki/Coffman%E2%80%93Graham_algorithm).
This would lead to not starting to evaluate a new topological-level expression
until at least one of the existing expressions is complete to minimize
contention for registers. This would also benefit the higher level search where
the evaluation order of many potential expressions could have a benefit given
the constraint on registers available.

##### Python Compiler Tools

Perhaps the state space evaluation or heuristic evaluation could reduce their
execution time via `numba` or `Cython`. Previous evaluations of this approach
didn't show a benefit proportional to the effort to maintain these
dependencies, but maybe this time it's different.
