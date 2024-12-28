package(default_visibility = ["//compiler:__pkg__"])

load("//py:defs.bzl", "PY_TEST_DEPS", "py_test_suite")
load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "ui-model-base",
    srcs = ["formak/ui/model_base.py"],
    imports = ["."],
    visibility = ["//visibility:private"],
    deps = [
        requirement("sympy"),
        "//py:exceptions",
    ],
)

py_library(
    name = "ui-model",
    srcs = ["formak/ui/model.py"],
    imports = ["."],
    visibility = ["//visibility:private"],
    deps = [
        requirement("sympy"),
        ":ui-model-base",
        "//py:exceptions",
    ],
)

py_library(
    name = "ui-state-machine",
    srcs = ["formak/ui/state_machine.py"],
    imports = ["."],
    visibility = ["//visibility:private"],
    deps = [
        requirement("scikit-learn"),  # this seems like an odd dependency, maybe some indirection could help here
    ],
)

py_library(
    name = "ui",
    srcs = ["formak/ui/__init__.py"],
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [
        ":ui-model",
        ":ui-state-machine",
    ],
)

py_test_suite(
    name = "ui-test",
    size = "small",
    srcs = [
        "test/.py",
    ],
    deps = [
        "//py:init-tree",
        ":ui",
    ] + PY_TEST_DEPS,
)
