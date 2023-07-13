# Generate C++ from AST

Author: Buck Baskin @bebaskin
Created: 2023-06-12
Updated: 2023-06-12
Parent Design: [designs/cpp_library_for_model_evaluation.md](../designs/cpp_library_for_model_evaluation.md)
Status: Merged

## Overview

FormaK aims to combine symbolic modeling for fast, efficient system modelling
with code generation to create performant code that is easy to use.

This design provides an extension to the fifth of the Five Keys
"C++ interfaces to support a variety of model uses" by reworking how C++
generation is done for easier extensions. After the Calibration design, a lot
of the code templates looked like:

```
        StateAndVariance
        ExtendedKalmanFilter::process_model(
            double dt,
            const StateAndVariance& input
            // clang-format off
{% if enable_calibration %}
            // clang-format on
            ,
            const Calibration& input_calibration
            // clang-format off
{% endif %}  // clang-format on
            // clang-format off
{% if enable_control %}
            // clang-format on
            ,
            const Control& input_control
            // clang-format off
{% endif %}  // clang-format on
        ) {
```

Instead of relying on increasingly intricate Jinja templating and managing
formatting via flagging clang-format on and off, I instead opted for another
approach: generate the code from an AST that approximated the Python AST. The
reason to go with something that approximates the Python AST is to have an
inspiration and a guide from an AST that has accumulated experience.

Afterwards, the code can look like:

```
        args = [
            Arg("double", "dt"),
            Arg("const StateAndVariance&", "input_state"),
        ]


        if enable_calibration:
            args.append(Arg("const Calibration&", "input_calibration"))
        if enable_control:
            args.append(Arg("const Control&", "input_control"))

        return FunctionDeclaration(
            "StateAndVariance",
            "process_model",
            args=args,
            modifier="",
        )
```

This approach isn't necessarily shorter, but it allows for replacing Jinja
templating with manipulating Python structures (primarily lists) in code. It
also generates cleaner code without droppings for clang-formatting

## Feature Tests

The feature tests for this were originally based on generating code to match
strings of examples purely in Python. Eventually, they were moved to C++
compilation to capture ensuring the overall feature generated valid C++.

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

### 2023-06-12

Overall, this design was a lot of manual work to translate over. I remain
optimistic that this translation will be worth it.

There are some areas where the learnings evolved over the project. Primarily,
this was the patterns for concisely and clearly manipulating the structures as
they were being implemented, especially args. Things evolved through:

- copying and pasting code
- wrapping the logic in functions, but still with repeated code (see `State_model`)
- finding the `standard_args` pattern (see the `ClassDef` for `{reading_type.typename}SensorModel`)
- In theory, this could go to filtering args based on zipping with an enable value, but I haven't gone to this yet (some of the other functional changes got quite long and full of parens)
