from formak.runtime import Storage, StorageLayout

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

    # Setup
    data.store(time, state, covariance, sensors)

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

    # Setup
    data.store(time, state, covariance, sensors)

    # After first time
    data.store(time + 1, state, covariance, sensors)

    assert list(data.scan()) == [
            (0, StorageLayout(time, state, covariance, sensors)),
            (1, StorageLayout(time + 1, state, covariance, sensors)),
            ]

def test_mid_insert():
    data = Storage()

    # Setup
    data.store()
    data.store()

    # Between two times
    data.store()

def test_exact_match_insert():
    data = Storage()

    # Setup
    data.store()

    # Exact same time
    data.store()

def test_inexact_match_insert():
    data = Storage()

    # Setup
    data.store()

    # Rounds to the same time
    data.store()

