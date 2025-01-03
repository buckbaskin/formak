package(default_visibility = ["//visibility:private"])

load("@pip_deps//:requirements.bzl", "requirement")
load("//py:defs.bzl", "PY_TEST_DEPS", "py_test_suite")

# TODO: split this into sub-packages
py_library(
    name = "runtime",
    srcs = ["runtime_py/runtime.py"],
    imports = ["."],
    visibility = ["//py:__pkg__"],
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
