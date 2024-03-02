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

    assert list(data.scan()) == [(time, StorageLayout(time, state, covariance, sensors))]

def test_pre_insert():
    data = Storage()

    # Setup
    data.store()

    # Before first time
    data.store()

def test_post_insert():
    data = Storage()

    # Setup
    data.store()

    # After first time
    data.store()


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

