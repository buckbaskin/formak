import pytest
from formak.runtime.storage import RollbackOptions, Storage, StorageLayout
from numpy.testing import assert_allclose

Row = StorageLayout


def assert_storage_close(actual, desired):
    assert len(actual) == len(desired)

    for left_value, right_value in zip(actual, desired):
        assert_allclose(left_value.time, right_value.time)
        assert left_value.state == right_value.state
        assert left_value.covariance == right_value.covariance
        assert left_value.sensors == right_value.sensors


def test_constructor():
    Storage()


def test_first_store():
    data = Storage()

    time = 1
    state = 2
    covariance = 3
    sensors = []

    data.store(time, state, covariance, sensors=sensors)

    assert len(data.data) == 1
    assert list(data.scan()) == [Row(time, state, covariance, None, sensors)]


def test_first_store_with_control():
    data = Storage()

    time = 1
    state = 2
    covariance = 3
    control = 4
    sensors = []

    data.store(time, state, covariance, control=control, sensors=sensors)

    assert len(data.data) == 1
    assert list(data.scan()) == [Row(time, state, covariance, control, sensors)]


def test_pre_store():
    data = Storage()

    time = 1
    state = 2
    covariance = 3
    sensors = []

    def setup():
        data.store(time, state, covariance, sensors=sensors)
        assert len(data.data) == 1

    setup()

    # Before first time
    data.store(time - 1, state, covariance, sensors=sensors)

    assert list(data.scan()) == [
        Row(time - 1, state, covariance, None, sensors),
        Row(time, state, covariance, None, sensors),
    ]


def test_pre_store_with_control():
    data = Storage()

    time = 1
    state = 2
    covariance = 3
    control = 4
    sensors = []

    def setup():
        data.store(time, state, covariance, control=control, sensors=sensors)
        assert len(data.data) == 1

    setup()

    # Before first time
    data.store(time - 1, state, covariance, control=control, sensors=sensors)

    assert list(data.scan()) == [
        Row(time - 1, state, covariance, control, sensors),
        Row(time, state, covariance, control, sensors),
    ]


def test_post_store():
    data = Storage()

    time = 1
    state = 2
    covariance = 3
    sensors = []

    def setup():
        data.store(time, state, covariance, sensors=sensors)
        assert len(data.data) == 1

    setup()

    # After first time
    data.store(time + 1, state, covariance, sensors=sensors)

    assert list(data.scan()) == [
        Row(time, state, covariance, None, sensors),
        Row(time + 1, state, covariance, None, sensors),
    ]


def test_post_store_with_control():
    data = Storage()

    time = 1
    state = 2
    covariance = 3
    control = 4
    sensors = []

    def setup():
        data.store(time, state, covariance, control=control, sensors=sensors)
        assert len(data.data) == 1

    setup()

    # After first time
    data.store(time + 1, state, covariance, control=control, sensors=sensors)

    assert list(data.scan()) == [
        Row(time, state, covariance, control, sensors),
        Row(time + 1, state, covariance, control, sensors),
    ]


def test_mid_store():
    data = Storage()

    time = 1
    state = 2
    covariance = 3
    sensors = []

    def setup():
        data.store(time - 1, state, covariance, sensors=sensors)
        data.store(time + 1, state, covariance, sensors=sensors)
        assert len(data.data) == 2

    setup()

    # Between two times
    data.store(time, state, covariance, sensors=sensors)

    assert list(data.scan()) == [
        Row(time - 1, state, covariance, None, sensors),
        Row(time, state, covariance, None, sensors),
        Row(time + 1, state, covariance, None, sensors),
    ]


def test_mid_store_with_control():
    data = Storage()

    time = 1
    state = 2
    covariance = 3
    control = 4
    sensors = []

    def setup():
        data.store(time - 1, state, covariance, control=control, sensors=sensors)
        data.store(time + 1, state, covariance, control=control, sensors=sensors)
        assert len(data.data) == 2

    setup()

    # Between two times
    data.store(time, state, covariance, control=control, sensors=sensors)

    assert list(data.scan()) == [
        Row(time - 1, state, covariance, control, sensors),
        Row(time, state, covariance, control, sensors),
        Row(time + 1, state, covariance, control, sensors),
    ]


def test_exact_match_store():
    data = Storage()

    time = 1
    state = 2
    covariance = 3
    sensors = ("A",)

    def setup():
        data.store(time, state, covariance, sensors=sensors)
        assert len(data.data) == 1

    setup()

    # Exact match in time
    data.store(time, -state, -covariance, sensors=("B", "C"))

    # Expect:
    #   - state overwriten
    #   - covariance overwriten
    #   - sensors extended
    assert list(data.scan()) == [
        Row(time, -state, -covariance, None, ["A", "B", "C"]),
    ]


def test_exact_match_store_with_control():
    data = Storage()

    time = 1
    state = 2
    covariance = 3
    control = 4
    sensors = ("A",)

    def setup():
        data.store(time, state, covariance, control=0, sensors=sensors)
        assert len(data.data) == 1

    setup()

    # Exact match in time
    data.store(time, -state, -covariance, control=control, sensors=("B", "C"))

    # Expect:
    #   - state overwriten
    #   - covariance overwriten
    #   - control overwriten
    #   - sensors extended
    assert list(data.scan()) == [
        Row(time, -state, -covariance, control, ["A", "B", "C"]),
    ]


def test_inexact_match_store_before_first():
    data = Storage(options=RollbackOptions(time_resolution=0.1))

    time = 1.1
    state = 2
    covariance = 3
    sensors = ("A",)

    def setup():
        data.store(time, state, covariance, sensors=sensors)
        data.store(time + 1, state, covariance, sensors=sensors)
        data.store(time + 2, state, covariance, sensors=sensors)
        assert len(data.data) == 3

    setup()

    data.store(time - 0.01, -state, -covariance, sensors=("B", "C"))

    # Expect:
    #   - original time preserved
    #   - state overwriten
    #   - covariance overwriten
    #   - sensors extended
    assert_storage_close(
        list(data.scan()),
        [
            Row(time, -state, -covariance, None, ["A", "B", "C"]),
            Row(time + 1, state, covariance, None, ["A"]),
            Row(time + 2, state, covariance, None, ["A"]),
        ],
    )


def test_inexact_match_store_before_first_with_control():
    data = Storage(options=RollbackOptions(time_resolution=0.1))

    time = 1.1
    state = 2
    covariance = 3
    control = 4
    sensors = ("A",)

    def setup():
        data.store(time, state, covariance, control=0, sensors=sensors)
        data.store(time + 1, state, covariance, control=0, sensors=sensors)
        data.store(time + 2, state, covariance, control=0, sensors=sensors)
        assert len(data.data) == 3

    setup()

    data.store(time - 0.01, -state, -covariance, control=control, sensors=("B", "C"))

    # Expect:
    #   - original time preserved
    #   - state overwriten
    #   - covariance overwriten
    #   - sensors extended
    assert_storage_close(
        list(data.scan()),
        [
            Row(time, -state, -covariance, control, ["A", "B", "C"]),
            Row(time + 1, state, covariance, 0, ["A"]),
            Row(time + 2, state, covariance, 0, ["A"]),
        ],
    )


def test_inexact_match_store_after_first():
    data = Storage(options=RollbackOptions(time_resolution=0.1))

    time = 1.1
    state = 2
    covariance = 3
    sensors = ("A",)

    def setup():
        data.store(time, state, covariance, sensors=sensors)
        data.store(time + 1, state, covariance, sensors=sensors)
        data.store(time + 2, state, covariance, sensors=sensors)
        assert len(data.data) == 3

    setup()

    data.store(time + 0.01, -state, -covariance, sensors=("B", "C"))

    # Expect:
    #   - original time preserved
    #   - state overwriten
    #   - covariance overwriten
    #   - control overwriten
    #   - sensors extended
    assert_storage_close(
        list(data.scan()),
        [
            Row(time, -state, -covariance, None, ["A", "B", "C"]),
            Row(time + 1, state, covariance, None, ["A"]),
            Row(time + 2, state, covariance, None, ["A"]),
        ],
    )


def test_inexact_match_store_after_first_with_control():
    data = Storage(options=RollbackOptions(time_resolution=0.1))

    time = 1.1
    state = 2
    covariance = 3
    control = 4
    sensors = ("A",)

    def setup():
        data.store(time, state, covariance, control=0, sensors=sensors)
        data.store(time + 1, state, covariance, control=0, sensors=sensors)
        data.store(time + 2, state, covariance, control=0, sensors=sensors)
        assert len(data.data) == 3

    setup()

    data.store(time + 0.01, -state, -covariance, control=control, sensors=("B", "C"))

    # Expect:
    #   - original time preserved
    #   - state overwriten
    #   - covariance overwriten
    #   - control overwriten
    #   - sensors extended
    assert_storage_close(
        list(data.scan()),
        [
            Row(time, -state, -covariance, control, ["A", "B", "C"]),
            Row(time + 1, state, covariance, 0, ["A"]),
            Row(time + 2, state, covariance, 0, ["A"]),
        ],
    )


def test_inexact_match_store_before_mid():
    data = Storage(options=RollbackOptions(time_resolution=0.1))

    time = 1.1
    state = 2
    covariance = 3
    sensors = ("A",)

    def setup():
        data.store(time - 1, state, covariance, sensors=sensors)
        data.store(time, state, covariance, sensors=sensors)
        data.store(time + 1, state, covariance, sensors=sensors)
        assert len(data.data) == 3

    setup()

    data.store(time - 0.01, -state, -covariance, sensors=("B", "C"))

    # Expect:
    #   - original time preserved
    #   - state overwriten
    #   - covariance overwriten
    #   - sensors extended
    assert_storage_close(
        list(data.scan()),
        [
            Row(time - 1, state, covariance, None, ["A"]),
            Row(time, -state, -covariance, None, ["A", "B", "C"]),
            Row(time + 1, state, covariance, None, ["A"]),
        ],
    )


def test_inexact_match_store_after_mid():
    data = Storage(options=RollbackOptions(time_resolution=0.1))

    time = 1.1
    state = 2
    covariance = 3
    sensors = ("A",)

    def setup():
        data.store(time - 1, state, covariance, sensors=sensors)
        data.store(time, state, covariance, sensors=sensors)
        data.store(time + 1, state, covariance, sensors=sensors)
        assert len(data.data) == 3

    setup()

    data.store(time + 0.01, -state, -covariance, sensors=("B", "C"))

    # Expect:
    #   - original time preserved
    #   - state overwriten
    #   - covariance overwriten
    #   - sensors extended
    assert_storage_close(
        list(data.scan()),
        [
            Row(time - 1, state, covariance, None, ["A"]),
            Row(time, -state, -covariance, None, ["A", "B", "C"]),
            Row(time + 1, state, covariance, None, ["A"]),
        ],
    )


def test_inexact_match_store_before_end():
    data = Storage(options=RollbackOptions(time_resolution=0.1))

    time = 1.1
    state = 2
    covariance = 3
    sensors = ("A",)

    def setup():
        data.store(time - 2, state, covariance, sensors=sensors)
        data.store(time - 1, state, covariance, sensors=sensors)
        data.store(time, state, covariance, sensors=sensors)
        assert len(data.data) == 3

    setup()

    data.store(time - 0.01, -state, -covariance, sensors=("B", "C"))

    # Expect:
    #   - original time preserved
    #   - state overwriten
    #   - covariance overwriten
    #   - sensors extended
    assert_storage_close(
        list(data.scan()),
        [
            Row(time - 2, state, covariance, None, ["A"]),
            Row(time - 1, state, covariance, None, ["A"]),
            Row(time, -state, -covariance, None, ["A", "B", "C"]),
        ],
    )


def test_inexact_match_store_after_end():
    data = Storage(options=RollbackOptions(time_resolution=0.1))

    time = 1.1
    state = 2
    covariance = 3
    sensors = ("A",)

    def setup():
        data.store(time - 2, state, covariance, sensors=sensors)
        data.store(time - 1, state, covariance, sensors=sensors)
        data.store(time, state, covariance, sensors=sensors)
        assert len(data.data) == 3

    setup()

    data.store(time + 0.01, -state, -covariance, sensors=("B", "C"))

    # Expect:
    #   - original time preserved
    #   - state overwriten
    #   - covariance overwriten
    #   - sensors extended
    assert_storage_close(
        list(data.scan()),
        [
            Row(time - 2, state, covariance, None, ["A"]),
            Row(time - 1, state, covariance, None, ["A"]),
            Row(time, -state, -covariance, None, ["A", "B", "C"]),
        ],
    )


def test_inexact_match_load_before_first():
    data = Storage(options=RollbackOptions(time_resolution=0.1))

    time = 1.1
    state = 2
    covariance = 3
    sensors = ("A",)

    def setup():
        data.store(time, state, covariance, sensors=sensors)
        data.store(time + 1, state, covariance, sensors=sensors)
        data.store(time + 2, state, covariance, sensors=sensors)
        assert len(data.data) == 3

    setup()

    load_time, _, _, _, _ = data.load(time - 0.01)
    assert_allclose(load_time, time)


def test_inexact_match_load_after_first():
    data = Storage(options=RollbackOptions(time_resolution=0.1))

    time = 1.1
    state = 2
    covariance = 3
    sensors = ("A",)

    def setup():
        data.store(time, state, covariance, sensors=sensors)
        data.store(time + 1, state, covariance, sensors=sensors)
        data.store(time + 2, state, covariance, sensors=sensors)
        assert len(data.data) == 3

    setup()

    load_time, _, _, _, _ = data.load(time + 0.01)
    assert_allclose(load_time, time)


def test_inexact_match_load_between():
    data = Storage(options=RollbackOptions(time_resolution=0.1))

    time = 1.1
    state = 2
    covariance = 3
    sensors = ("A",)

    def setup():
        data.store(time - 1, state, covariance, sensors=sensors)
        data.store(time, state, covariance, sensors=sensors)
        data.store(time + 1, state, covariance, sensors=sensors)
        assert len(data.data) == 3

    setup()

    load_time, _, _, _, _ = data.load(time - 0.5)
    assert_allclose(load_time, time - 1)


def test_inexact_match_load_before_mid():
    data = Storage(options=RollbackOptions(time_resolution=0.1))

    time = 1.1
    state = 2
    covariance = 3
    sensors = ("A",)

    def setup():
        data.store(time - 1, state, covariance, sensors=sensors)
        data.store(time, state, covariance, sensors=sensors)
        data.store(time + 1, state, covariance, sensors=sensors)
        assert len(data.data) == 3

    setup()

    load_time, _, _, _, _ = data.load(time - 0.01)
    assert_allclose(load_time, time - 1)


def test_inexact_match_load_after_mid():
    data = Storage(options=RollbackOptions(time_resolution=0.1))

    time = 1.1
    state = 2
    covariance = 3
    sensors = ("A",)

    def setup():
        data.store(time - 1, state, covariance, sensors=sensors)
        data.store(time, state, covariance, sensors=sensors)
        data.store(time + 1, state, covariance, sensors=sensors)
        assert len(data.data) == 3

    setup()

    load_time, _, _, _, _ = data.load(time + 0.01)
    assert_allclose(load_time, time)


def test_inexact_match_load_before_end():
    data = Storage(options=RollbackOptions(time_resolution=0.1))

    time = 1.1
    state = 2
    covariance = 3
    sensors = ("A",)

    def setup():
        data.store(time - 2, state, covariance, sensors=sensors)
        data.store(time - 1, state, covariance, sensors=sensors)
        data.store(time, state, covariance, sensors=sensors)
        assert len(data.data) == 3

    setup()

    load_time, _, _, _, _ = data.load(time - 0.01)
    assert_allclose(load_time, time - 1)


def test_inexact_match_load_after_end():
    data = Storage(options=RollbackOptions(time_resolution=0.1))

    time = 1.1
    state = 2
    covariance = 3
    sensors = ("A",)

    def setup():
        data.store(time - 2, state, covariance, sensors=sensors)
        data.store(time - 1, state, covariance, sensors=sensors)
        data.store(time, state, covariance, sensors=sensors)
        assert len(data.data) == 3

    setup()

    load_time, _, _, _, _ = data.load(time + 0.01)
    assert_allclose(load_time, time)


def test_scans():
    data = Storage(options=RollbackOptions(time_resolution=0.1))

    time = 1.1
    state = 2
    covariance = 3
    sensors = ("A",)

    def setup():
        data.store(time - 1, state, covariance, sensors=sensors)
        data.store(time, state, covariance, sensors=sensors)
        data.store(time + 1, state, covariance, sensors=sensors)
        assert len(data.data) == 3

    setup()

    scan_all = list(data.scan(0.0, 3.0))
    assert_storage_close(
        scan_all,
        [
            Row(time - 1, state, covariance, None, ["A"]),
            Row(time, state, covariance, None, ["A"]),
            Row(time + 1, state, covariance, None, ["A"]),
        ],
    )

    scan_early = list(data.scan(0.0, 1.2))
    assert_storage_close(
        scan_early,
        [
            Row(time - 1, state, covariance, None, ["A"]),
            Row(time, state, covariance, None, ["A"]),
        ],
    )

    scan_late = list(data.scan(0.5, 2.9))
    assert_storage_close(
        scan_late,
        [
            Row(time, state, covariance, None, ["A"]),
            Row(time + 1, state, covariance, None, ["A"]),
        ],
    )

    scan_first = list(data.scan(0.05, 0.15))
    assert_storage_close(
        scan_first,
        [
            Row(time - 1, state, covariance, None, ["A"]),
        ],
    )

    scan_last = list(data.scan(2.05, 2.15))
    assert_storage_close(
        scan_last,
        [
            Row(time + 1, state, covariance, None, ["A"]),
        ],
    )

    scan_empty = list(data.scan(4.0, 4.1))
    assert_storage_close(
        scan_empty,
        [],
    )

    with pytest.raises(TypeError):
        list(data.scan(5.0))
