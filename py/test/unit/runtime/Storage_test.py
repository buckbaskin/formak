from formak.runtime import RollbackOptions, Storage, StorageLayout


def test_constructor():
    Storage()


def test_first_insert():
    data = Storage()

    time = 1
    state = 2
    covariance = 3
    sensors = []

    data.store(time, state, covariance, sensors)

    assert list(data.scan()) == [(0, StorageLayout(time, state, covariance, sensors))]


def test_pre_insert():
    data = Storage()

    time = 1
    state = 2
    covariance = 3
    sensors = []

    def setup():
        data.store(time, state, covariance, sensors)

    setup()

    # Before first time
    data.store(time - 1, state, covariance, sensors)

    assert list(data.scan()) == [
        (0, StorageLayout(time - 1, state, covariance, sensors)),
        (1, StorageLayout(time, state, covariance, sensors)),
    ]


def test_post_insert():
    data = Storage()

    time = 1
    state = 2
    covariance = 3
    sensors = []

    def setup():
        data.store(time, state, covariance, sensors)

    setup()

    # After first time
    data.store(time + 1, state, covariance, sensors)

    assert list(data.scan()) == [
        (0, StorageLayout(time, state, covariance, sensors)),
        (1, StorageLayout(time + 1, state, covariance, sensors)),
    ]


def test_mid_insert():
    data = Storage()

    time = 1
    state = 2
    covariance = 3
    sensors = []

    def setup():
        data.store(time - 1, state, covariance, sensors)
        data.store(time + 1, state, covariance, sensors)

    setup()

    # Between two times
    data.store(time, state, covariance, sensors)

    assert list(data.scan()) == [
        (0, StorageLayout(time - 1, state, covariance, sensors)),
        (1, StorageLayout(time, state, covariance, sensors)),
        (2, StorageLayout(time + 1, state, covariance, sensors)),
    ]


def test_exact_match_insert():
    data = Storage()

    time = 1
    state = 2
    covariance = 3
    sensors = ["A"]

    def setup():
        data.store(time, state, covariance, sensors)

    setup()

    # Exact match in time
    data.store(time, -state, -covariance, ["B", "C"])

    # Expect:
    #   - state overwriten
    #   - covariance overwriten
    #   - sensors extended
    assert list(data.scan()) == [
        (0, StorageLayout(time, -state, -covariance, ["A", "B", "C"])),
    ]


def test_inexact_match_insert():
    data = Storage(options=RollbackOptions(time_resolution=0.1))

    time = 1.01
    state = 2
    covariance = 3
    sensors = ["A"]

    def setup():
        data.store(time, state, covariance, sensors)

    setup()

    # Exact match in time
    data.store(time + 0.02, -state, -covariance, ["B", "C"])

    # TODO(buck): should the time resolution affect how the time is stored? e.g. store rounded to the first value in the time window?

    # Expect:
    #   - original time preserved
    #   - state overwriten
    #   - covariance overwriten
    #   - sensors extended
    assert list(data.scan()) == [
        (0, StorageLayout(time, -state, -covariance, ["A", "B", "C"])),
    ]
