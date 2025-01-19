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
)

py_library(
    name = "model",
    srcs = ["backend_py/model.py"],
    imports = ["."],
    visibility = [
        "//py:__pkg__",
    ],
)
