load("//py:defs.bzl", "PY_TEST_DEPS", "py_test_suite")
load("@pip_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:private"])

# Opportunity to split this into more specific classes
py_library(
    name = "stamped-reading",
    srcs = ["formak/runtime/stamped_reading.py"],
    imports = ["."],
)

py_library(
    name = "sensor-model",
    srcs = ["formak/runtime/sensor_model.py"],
    imports = ["."],
    deps = [
        "//common:named-covariance",
        "//common:named-vector",
        "//compiler/py:basic-block",
    ],
)

py_library(
    name = "managed-filter",
    srcs = ["formak/runtime/managed_filter.py"],
    imports = ["."],
    deps = [":stamped-reading"],
)

py_library(
    name = "sklearn",
    srcs = ["formak/runtime/sklearn.py"],
    imports = ["."],
    visibility = ["//formak:__pkg__"],
    deps = [
        requirement("scikit-learn"),
        ":sensor-model",
    ],
)

py_test_suite(
    name = "runtime-test",
    timeout = "moderate",
    srcs = glob(
        [
            "test/*.py",
        ],
        allow_empty = False,
    ),
    deps = [
        "//py:init-tree",
        ":managed-filter",
    ] + PY_TEST_DEPS,
)
