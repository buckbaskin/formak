load("//py:defs.bzl",  "py_test_suite", "PY_TEST_DEPS")
load("@pip_deps//:requirements.bzl", "requirement")

package(default_visibility = ["//visibility:private"])

# Opportunity to split this into more specific classes
py_library(
    name = "runtime",
    srcs = ["formak/runtime.py"],
    imports = ["."],
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
        ":runtime",
        "//frontend:ui",
    ] + PY_TEST_DEPS,
)
