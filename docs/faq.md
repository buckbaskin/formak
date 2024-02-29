# Frequently Asked Questions

## How do I get started creating a model?

See [Getting Started](getting-started) for instructions on how to create your
first model

## Where can I find introductory examples?

See the [demo/](https://github.com/buckbaskin/formak/tree/main/demo) directory
for introductory examples.

## How would you recommend to handle different rates and possibly lack of exact synchronization for the models which FormaK produces?

If you pass in the sensors and control information, the `ManagedFilter` will
take care of the rest, including synchronizing your sensor readings. This
reduces code complexity for you and opens the opportunity to adopt additional
benefits as well.

In the
[current version](https://github.com/buckbaskin/formak/commit/f7b5267ae81494b4327d66f3152f915d0fa4c5c9)
of the library, there is manual work required to perform synchronization. The
outline of the algorithm I'd recommend is:

1. Maintain a clock reference for publishing time
2. Queue inputs as they arrive (e.g. as ROS messages from subscribers), sorted by time
3. At the desired output rate, take the front of the message queue that are at or before the desired reference time
4. Pass that list/vector as input to the FormaK `ManagedFilter` `.tick` method (`managed.tick(reference_time, list_of_messages)`) and the `ManagedFilter` will process the messages according to their time stamp using the process model to shift in time and align with the stamped time
5. The output from the `tick` function will be the estimate of the filter for all of the messages up to the reference time

[Example usage of the `ManagedFilter` class](https://github.com/buckbaskin/formak/blob/f7b5267ae81494b4327d66f3152f915d0fa4c5c9/py/test/unit/runtime/ManagedFilter_no_calibration_test.py#L218)

If `ManagedFilter` isn't serving your use case, please reach out to
formak.open.source@gmail.com or submit an issue via
[Github](https://github.com/buckbaskin/formak/issues). I'd love to learn more
about your use case and see if it's something that can be supported by the
library.

