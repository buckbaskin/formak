from formak.runtime import RollbackOptions, Storage, StorageLayout
from numpy.testing import assert_allclose

Row = StorageLayout


def assert_storage_close(actual, desired):
    assert len(actual) == len(desired)

    for (left_idx, left_value), (right_idx, right_value) in zip(actual, desired):
        assert left_idx == right_idx

        assert_allclose(left_value.time, right_value.time)
        assert left_value.state == right_value.state
        assert left_value.covariance == right_value.covariance
        assert left_value.sensors == right_value.sensors


def test_constructor():
    Storage()


def test_first_insert():
    data = Storage()

    time = 1
    state = 2
    covariance = 3
    sensors = []

    data.store(time, state, covariance, sensors)

    assert len(data.data) == 1
    assert list(data.scan()) == [(0, Row(time, state, covariance, sensors))]


def test_pre_insert():
    data = Storage()

    time = 1
    state = 2
    covariance = 3
    sensors = []

    def setup():
        data.store(time, state, covariance, sensors)
        assert len(data.data) == 1

    setup()

    # Before first time
    data.store(time - 1, state, covariance, sensors)

    assert list(data.scan()) == [
        (0, Row(time - 1, state, covariance, sensors)),
        (1, Row(time, state, covariance, sensors)),
    ]


def test_post_insert():
    data = Storage()

    time = 1
    state = 2
    covariance = 3
    sensors = []

    def setup():
        data.store(time, state, covariance, sensors)
        assert len(data.data) == 1

    setup()

    # After first time
    data.store(time + 1, state, covariance, sensors)

    assert list(data.scan()) == [
        (0, Row(time, state, covariance, sensors)),
        (1, Row(time + 1, state, covariance, sensors)),
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
        assert len(data.data) == 2

    setup()

    # Between two times
    data.store(time, state, covariance, sensors)

    assert list(data.scan()) == [
        (0, Row(time - 1, state, covariance, sensors)),
        (1, Row(time, state, covariance, sensors)),
        (2, Row(time + 1, state, covariance, sensors)),
    ]


def test_exact_match_insert():
    data = Storage()

    time = 1
    state = 2
    covariance = 3
    sensors = ("A",)

    def setup():
        data.store(time, state, covariance, sensors)
        assert len(data.data) == 1

    setup()

    # Exact match in time
    data.store(time, -state, -covariance, ("B", "C"))

    # Expect:
    #   - state overwriten
    #   - covariance overwriten
    #   - sensors extended
    assert list(data.scan()) == [
        (0, Row(time, -state, -covariance, ["A", "B", "C"])),
    ]


def test_inexact_match_insert_before_first():
    data = Storage(options=RollbackOptions(time_resolution=0.1))

    time = 1.1
    state = 2
    covariance = 3
    sensors = ("A",)

    def setup():
        data.store(time, state, covariance, sensors)
        data.store(time + 1, state, covariance, sensors)
        data.store(time + 2, state, covariance, sensors)
        assert len(data.data) == 3

    setup()

    data.store(time - 0.01, -state, -covariance, ("B", "C"))

    # TODO(buck): should the time resolution affect how the time is stored? e.g. store rounded to the first value in the time window?

    # Expect:
    #   - original time preserved
    #   - state overwriten
    #   - covariance overwriten
    #   - sensors extended
    assert_storage_close(
        list(data.scan()),
        [
            (0, Row(time, -state, -covariance, ["A", "B", "C"])),
            (1, Row(time + 1, state, covariance, ["A"])),
            (2, Row(time + 2, state, covariance, ["A"])),
        ],
    )


def test_inexact_match_insert_after_first():
    data = Storage(options=RollbackOptions(time_resolution=0.1))

    time = 1.1
    state = 2
    covariance = 3
    sensors = ("A",)

    def setup():
        data.store(time, state, covariance, sensors)
        data.store(time + 1, state, covariance, sensors)
        data.store(time + 2, state, covariance, sensors)
        assert len(data.data) == 3

    setup()

    data.store(time + 0.01, -state, -covariance, ("B", "C"))

    # TODO(buck): should the time resolution affect how the time is stored? e.g. store rounded to the first value in the time window?

    # Expect:
    #   - original time preserved
    #   - state overwriten
    #   - covariance overwriten
    #   - sensors extended
    assert_storage_close(
        list(data.scan()),
        [
            (0, Row(time, -state, -covariance, ["A", "B", "C"])),
            (1, Row(time + 1, state, covariance, ["A"])),
            (2, Row(time + 2, state, covariance, ["A"])),
        ],
    )


def test_inexact_match_insert_before_mid():
    data = Storage(options=RollbackOptions(time_resolution=0.1))

    time = 1.1
    state = 2
    covariance = 3
    sensors = ("A",)

    def setup():
        data.store(time - 1, state, covariance, sensors)
        data.store(time, state, covariance, sensors)
        data.store(time + 1, state, covariance, sensors)
        assert len(data.data) == 3

    setup()

    data.store(time - 0.01, -state, -covariance, ("B", "C"))

    # TODO(buck): should the time resolution affect how the time is stored? e.g. store rounded to the first value in the time window?

    # Expect:
    #   - original time preserved
    #   - state overwriten
    #   - covariance overwriten
    #   - sensors extended
    assert_storage_close(
        list(data.scan()),
        [
            (0, Row(time - 1, state, covariance, ["A"])),
            (1, Row(time, -state, -covariance, ["A", "B", "C"])),
            (2, Row(time + 1, state, covariance, ["A"])),
        ],
    )


def test_inexact_match_insert_after_mid():
    data = Storage(options=RollbackOptions(time_resolution=0.1))

    time = 1.1
    state = 2
    covariance = 3
    sensors = ("A",)

    def setup():
        data.store(time - 1, state, covariance, sensors)
        data.store(time, state, covariance, sensors)
        data.store(time + 1, state, covariance, sensors)
        assert len(data.data) == 3

    setup()

    data.store(time + 0.01, -state, -covariance, ("B", "C"))

    # TODO(buck): should the time resolution affect how the time is stored? e.g. store rounded to the first value in the time window?

    # Expect:
    #   - original time preserved
    #   - state overwriten
    #   - covariance overwriten
    #   - sensors extended
    assert_storage_close(
        list(data.scan()),
        [
            (0, Row(time - 1, state, covariance, ["A"])),
            (1, Row(time, -state, -covariance, ["A", "B", "C"])),
            (2, Row(time + 1, state, covariance, ["A"])),
        ],
    )


def test_inexact_match_insert_before_end():
    data = Storage(options=RollbackOptions(time_resolution=0.1))

    time = 1.1
    state = 2
    covariance = 3
    sensors = ("A",)

    def setup():
        data.store(time - 2, state, covariance, sensors)
        data.store(time - 1, state, covariance, sensors)
        data.store(time, state, covariance, sensors)
        assert len(data.data) == 3

    setup()

    data.store(time - 0.01, -state, -covariance, ("B", "C"))

    # TODO(buck): should the time resolution affect how the time is stored? e.g. store rounded to the first value in the time window?

    # Expect:
    #   - original time preserved
    #   - state overwriten
    #   - covariance overwriten
    #   - sensors extended
    assert_storage_close(
        list(data.scan()),
        [
            (0, Row(time - 2, state, covariance, ["A"])),
            (1, Row(time - 1, state, covariance, ["A"])),
            (2, Row(time, -state, -covariance, ["A", "B", "C"])),
        ],
    )


def test_inexact_match_insert_after_end():
    data = Storage(options=RollbackOptions(time_resolution=0.1))

    time = 1.1
    state = 2
    covariance = 3
    sensors = ("A",)

    def setup():
        data.store(time - 2, state, covariance, sensors)
        data.store(time - 1, state, covariance, sensors)
        data.store(time, state, covariance, sensors)
        assert len(data.data) == 3

    setup()

    data.store(time + 0.01, -state, -covariance, ("B", "C"))

    # TODO(buck): should the time resolution affect how the time is stored? e.g. store rounded to the first value in the time window?

    # Expect:
    #   - original time preserved
    #   - state overwriten
    #   - covariance overwriten
    #   - sensors extended
    assert_storage_close(
        list(data.scan()),
        [
            (0, Row(time - 2, state, covariance, ["A"])),
            (1, Row(time - 1, state, covariance, ["A"])),
            (2, Row(time, -state, -covariance, ["A", "B", "C"])),
        ],
    )
