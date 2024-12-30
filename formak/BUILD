package(default_visibility = ["//visibility:public"])

load("//py:defs.bzl", "PY_TEST_DEPS", "py_test_suite")
load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "formak",
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [
        # ":cpp",
        # ":python",
        # ":reference-models", # including this breaks feature tests?
        # ":ui", # including this breaks feature tests?
        "//compiler",
        "//backend",
        # ":runtime",
    ],
)

py_library(
    name = "ui-model-base",
    srcs = ["formak/ui/model_base.py"],
    imports = ["."],
    visibility = [
        "//compiler/py:__pkg__",  # TODO antipattern, //compiler should not depend on this, should be vis:private
    ],
    deps = [
        requirement("sympy"),
        "//py:exceptions",
    ],
)

py_library(
    name = "ui-model",
    srcs = ["formak/ui/model.py"],
    imports = ["."],
    visibility = [
        "//featuretests/scikit_learn_integration:__pkg__",  # TODO antipattern, should be vis:private
    ],
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
        "//compiler/py:config",
        "//runtime/py:sklearn",
        requirement("scikit-learn"),  # this seems like an odd dependency, maybe some indirection could help here
    ],
)

py_library(
    name = "ui",
    srcs = [],
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
        "test/ui_test.py",
    ],
    deps = [
        "//py:init-tree",
        ":ui",
    ] + PY_TEST_DEPS,
)

py_library(
    name = "reference-models",
    srcs = glob(
        ["formak/reference_models/*.py"],
        allow_empty = False,
    ),
    imports = ["."],
    deps = [
        "//formak:ui",
        requirement("sympy"),
    ],
)
