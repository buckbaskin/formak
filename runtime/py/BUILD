load("//py:defs.bzl", "PY_TEST_DEPS", "py_test_suite")
load("@pip_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:private"])

# Opportunity to split this into more specific classes
py_library(
    name = "stamped-reading",
    srcs = ["formak/runtime/stamped_reading.py"],
)

py_library(
    name = "managed-filter",
    srcs = ["formak/runtime/managed_filter.py"],
    deps = [":stamped-reading"],
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
        "//frontend:ui",
    ] + PY_TEST_DEPS,
)
