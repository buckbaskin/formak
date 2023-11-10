import numpy as np


def generate_data(innovation, *, samples=100):
    rng = np.random.rng()

    mean = 0
    stddev = 1
    data = rng.normal(mean, stddev, (samples, 1))

    even = True

    for i in range(samples):
        if i % 10:
            data[i, 0] = innovation + 1

    return data
