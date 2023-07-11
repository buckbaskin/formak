from datetime import datetime


def microbenchmark(function, input_iterable):
    """
    Time the lambda function for each input
    """
    if len(input_iterable) == 0:
        raise ValueError("Input Iterable must have at least one value")

    times = []
    for state_vector in input_iterable:
        start_time = datetime.now()

        for _ in range(10):
            function(state_vector)

        end_time = datetime.now()
        dt = end_time - start_time
        times.append(dt)

    assert len(times) > 0

    return times
