package(default_visibility = ["//visibility:public"])

load("//py:defs.bzl", "PY_TEST_DEPS", "py_test_suite")
load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "named-array-base",
    srcs = ["formak/common/named_array_base.py"],
    imports = ["."],
    deps = [
    ],
)

py_library(
    name = "named-vector",
    srcs = ["formak/common/named_vector.py"],
    imports = ["."],
    deps = [
    ],
)

py_library(
    name = "named-covariance",
    srcs = ["formak/common/named_covariance.py"],
    imports = ["."],
    deps = [
    ],
)

py_library(
    name = "plot",
    srcs = ["formak/common/plot.py"],
    imports = ["."],
    deps = [
    ],
)

py_test_suite(
    name = "ui-test",
    size = "small",
    srcs = [
        "test/ui_test.py",
    ],
    deps = [
        "//py:init-tree",
        ":ui",
    ] + PY_TEST_DEPS,
)
