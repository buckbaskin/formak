# FormaK Managed Runtime

Author: Buck Baskin @bebaskin
Created: 2023-07-13
Updated: 2023-07-15
Parent Design: [designs/cpp_library_for_model_evaluation.md](../designs/cpp_library_for_model_evaluation.md)
See Also: [designs/python_library_for_model_evaluation.md](../designs/python_library_for_model_evaluation.md)
Status: 2. Write a feature test(s)

## Overview

FormaK aims to combine symbolic modeling for fast, efficient system modelling
with code generation to create performant code that is easy to use.

The Five Key Elements the library provides to achieve this user experience are:
1. Python Interface to define models
2. Python implementation of the model and supporting tooling
3. Integration to scikit-learn to leverage the model selection and parameter tuning functions
4. C++ and Python to C++ interoperability for performance
5. C++ interfaces to support a variety of model uses

This design provides an extension to the second of the Five Keys "Python
implementation of the model..." and the fifth of the Five Keys "C++ interfaces
to support a variety of model uses" to support easier to use filters. The
current implementation provides some of the math for the EKF (process update,
sensor update) which can be used in a flexible manner, but don't necessarily
meet the easy to use goal for the project.

This feature aims to make the library easier to use by providing a managed
filter with a single interface.

This feature also has a forward looking benefit: by setting up the Managed
Filter structure now, it will be easier to add "netcode" features, logging and
other runtime benefits within the context of a unified Managed Filter.

## Solution Approach

The primary class of interest will be a new class `ManagedFilter` (in both C++
and Python).

This class will have a member function `tick` that will be the primary user
facing to the rest of the filter logic.

    // No sensor reading, process update only
    tick(time)
    tick(time, [])

    // One sensor reading
    tick(time, [reading])

    // Multiple sensor readings
    tick(time, [reading1, reading2, reading3])

The function will return the current state and variance of the filter after
processing the tick. By using the `ManagedFilter` wrapper, the user doesn't
need to worry about tracking states, variances, times, the process model (or
models in the future), sensor models. Instead the user just passes in the
desired output time and any new information from sensors and gets the result.

### Goals for the `ManagedFilter` class

Goals for the `ManagedFilter` class:
1. Minimal overhead on top of the underlying filter
2. Easy to use
3. Hard to misuse
4. No code generation
5. Compatible with multiple potential filters

FormaK's high level goals also apply to this class design.

> FormaK aims ... to create performant code that is easy to use [in that order]

For this design, "performance" will be focused on minimizing unnecessary copies
and allocations during filter management via clean implementation. At a future
point, profiling will provide data to inform evolutions of the design
internals.

This means that the primary driving principle for the design will be ease of
use.

A sub-goal of `Easy to use` is making it hard to misuse. If the code compiles,
the user should be confident that the underlying filter is working as expected.

Another sub-goal of `Easy to use`: No code generation. If the interface to the
class changed based on code generation, it could make it harder to understand
how to use it correctly. The existing filters take on a large amount of
complexity to make their use both powerful and avoid the cost of unused
features via code generation. The `ManagedFilter` shouldn't pass through this
complexity to the user.

This last goal is primarily forward looking. The `ManagedFilter` should be
compatible with multiple filter implementations for two use cases. First, as
FormaK evolves it may make sense to have different filter types being managed
(e.g. EKF and UKF or multiple EKF for a multiple model filter). Second, users
could provide their own variation of a filter implementation with features not
included in FormaK filters. If the `ManagedFilter` can support this, the user
can still get the benefits of the runtime even if they don't want the specific
filter details. This second benefit should, with marginal effort, fall out as a
consequence of the first use case.

### Managing Computation

The `tick` member function will manage the computation of the underlying
filter. The basic underlying algorithm will be:

    def tick(output_time, readings):
        for sensor_reading in readings:
            process_update(sensor_reading.time)
            sensor_update(sensor_reading)

        process_update(output_time)

The initial approach for the Filter will take inspiration from delay-based
netcode and hold back its state to the last sensor time. This should ensure
maximal information gain by minimizing uncertainty gained due to rolling back
in time. This makes the underlying algorithm only slightly more complicated:

    def tick(output_time, readings):
        for sensor_reading in readings:
            self.state = process_update(self.state, sensor_reading.time)
            self.state = sensor_update(self.state, sensor_reading)

        return process_update(self.state, output_time)

Note that the last process update is returned but doesn't update the state of
the model.

### Managing Memory

The `ManagedFilter` class will own the memory for the underlying computation;
however, the layout for the underlying computation will need to be provided by
the generated computation as a struct.

This is an expansion of the responsibility for the underlying filter
implementer (also FormaK at this time), but should help separate concerns.
- State is owned by the `ManagedFilter`
- Stateless mathematical logic is owned by the filter implementation

This may also require a refactor in how the underlying filter implementation is
defined. Currently, it is done as a mix of free functions. To support the move
to the `ManagedFilter` the generation may move to bundling these functions into
a class to make it easier to tag the filter implementation with metadata (at
minimum but not limited to the memory layout).

### Configuration

The `ManagedFilter` constructor should accept a struct `ManagedFilterOptions`
to allow for user selection of different features. This could be omitted if
there are no options for the current implementation. The options will certainly
come in the future.

One thing that isn't clear is how these options will interact with the options
specified by users during filter generation.

Perhaps they are a second set of configuration related to filter management
that is only relevant to the `ManagedFilter`?

That doesn't hold water at this time because model fitting behavior will depend
on how the `ManagedFilter` would be run (e.g. does it perform out of order
sensor updates?) but I don't have a good answer at design time.

## Feature Tests

The feature tests for this design will focus on the tick interface in a few
combinations:

- No sensor updates (process only)
- One sensor update
- Multiple sensor updates

The goal for the feature tests is to focus on the filter management and its
ease of use, not the EKF math itself, so assertions will focus on time
management and broad trends in state and variance for a simple model where it's
easy to calculate model evolution by hand.

## Road Map and Process

1. Write a design
2. Write a feature test(s)
3. Build a simple prototype
4. Pass feature tests
5. Refactor/cleanup
6. Build an instructive prototype (e.g. something that looks like the project vision but doesn't need to be the full thing)
7. Add unit testing, etc
8. Refactor/cleanup
9. Write up successes, retro of what changed (so I can check for this in future designs)

## Post Review
