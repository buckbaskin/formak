import numpy as np


def generate_data(innovation, *, samples=100):
    rng = np.random.default_rng(seed=5)

    mean = 0
    stddev = 1
    data = rng.normal(loc=mean, scale=stddev, size=(samples, 1))

    even = True

    for i in range(samples):
        if i % 10 == 3:
            data[i, 0] = innovation + 1

    return data
