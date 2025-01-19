package(default_visibility = ["//visibility:private"])

load("@pip_deps//:requirements.bzl", "requirement")
load("//py:defs.bzl", "PY_TEST_DEPS", "py_test_suite")

py_library(
    name = "named-array-base",
    srcs = ["backend_py/named_array_base.py"],
    imports = ["."],
)

py_library(
    name = "named-vector",
    srcs = ["backend_py/named_vector.py"],
    imports = ["."],
    visibility = [
        "//backend_cpp:__pkg__",
        "//py:__pkg__",
    ],
    deps = [":named-array-base"],
)

py_test_suite(
    name = "named-vector-test",
    srcs =
        ["test/named_vector_test.py"],
    deps = [
        ":named-vector",
    ] + PY_TEST_DEPS,
)

py_library(
    name = "named-covariance",
    srcs = ["backend_py/named_covariance.py"],
    imports = ["."],
    visibility = [
        "//backend_cpp:__pkg__",
        "//py:__pkg__",
    ],
    deps = [":named-array-base"],
)

py_test_suite(
    name = "named-covariance-test",
    srcs =
        ["test/named_covariance_test.py"],
    deps = [
        ":named-covariance",
    ] + PY_TEST_DEPS,
)

py_library(
    name = "config",
    srcs = ["backend_py/config.py"],
    imports = ["."],
    visibility = [
        "//py:__pkg__",
    ],
)

py_library(
    name = "basic-block",
    srcs = ["backend_py/basic_block.py"],
    imports = ["."],
    visibility = [
        "//py:__pkg__",
    ],
    deps = [
        requirement("sympy"),
        requirement("scipy"),
    ],
)

py_library(
    name = "model",
    srcs = ["backend_py/model.py"],
    imports = ["."],
    visibility = [
        "//py:__pkg__",
    ],
    deps = [
        ":named-vector",
        "//py:exceptions",
    ],
)

py_library(
    name = "sensor-model",
    srcs = ["backend_py/sensor_model.py"],
    imports = ["."],
    visibility = [
        "//py:__pkg__",
    ],
)

py_library(
    name = "extended-kalman-filter",
    srcs = ["backend_py/extended_kalman_filter.py"],
    imports = ["."],
    visibility = [
        "//py:__pkg__",
    ],
    deps = [
        ":basic-block",
        ":model",
        ":named-covariance",
        ":sensor-model",
        "//problemdefinition:ui-model-base",
    ],
)

py_library(
    name = "compile-model",
    srcs = ["backend_py/compile_model.py"],
    imports = ["."],
    visibility = [
        "//py:__pkg__",
    ],
)

py_library(
    name = "compile-ekf",
    srcs = ["backend_py/compile_ekf.py"],
    imports = ["."],
    visibility = [
        "//py:__pkg__",
        "//runtime_py:__pkg__",
    ],
    deps = [
        ":config",
        ":extended-kalman-filter",
        "//frontend:model-validation",
    ],
)

# TODO: this should be runtime
py_library(
    name = "sklearn",
    srcs = ["backend_py/sklearn.py"],
    imports = ["."],
    visibility = [
        "//problemdefinition:__pkg__",
        "//py:__pkg__",
    ],
    deps = [
        ":compile-ekf",
        requirement("scikit-learn"),
    ],
)

py_test_suite(
    name = "innovation-test",
    srcs = [
        "test/innovation_test.py",
    ],
    deps = [
        ":compile-ekf",
        "//problemdefinition:model",
    ] + PY_TEST_DEPS,
)

py_test_suite(
    name = "calibration-test",
    srcs = [
        "test/calibration_test.py",
    ],
    deps = [
        ":compile-ekf",
        ":compile-model",
        "//problemdefinition:model",
    ] + PY_TEST_DEPS,
)

py_test_suite(
    name = "extended-kalman-filter-test",
    srcs = [
        "test/ExtendedKalmanFilter_test.py",
    ],
    deps = [
        ":extended-kalman-filter",
        ":compile-ekf",
        "//problemdefinition:model",
    ] + PY_TEST_DEPS,
)
