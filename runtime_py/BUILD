package(default_visibility = ["//visibility:private"])

load("@pip_deps//:requirements.bzl", "requirement")
load("//py:defs.bzl", "py_test_suite", "PY_TEST_DEPS")

# TODO: split this into sub-packages
py_library(
    name = "runtime",
    srcs = ["runtime_py/runtime.py"],
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
        # ":init-tree",
        ":runtime",
        "//problemdefinition:model",
        "//py:python",
    ] + PY_TEST_DEPS,
)

