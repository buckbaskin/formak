package(default_visibility = ["//visibility:private"])

load("@pip_deps//:requirements.bzl", "requirement")
load("//py:defs.bzl", "PY_TEST_DEPS", "py_test_suite")

py_library(
    name = "stamped-reading",
    srcs = ["runtime_py/stamped_reading.py"],
    imports = ["."],
    visibility = ["//py:__pkg__"],
)

py_library(
    name = "managed-filter",
    srcs = ["runtime_py/managed_filter.py"],
    imports = ["."],
    visibility = ["//py:__pkg__"],
)

py_test_suite(
    name = "managed-filter-test",
    timeout = "moderate",
    srcs = glob(
        [
            "test/ManagedFilter*.py",
        ],
        allow_empty = False,
    ),
    deps = [
        ":managed-filter",
        ":stamped-reading",
        "//backend_py:compile-ekf",
        "//problemdefinition:model",
    ] + PY_TEST_DEPS,
)

py_library(
    name = "plotting",
    srcs = ["runtime_py/plotting.py"],
    imports = ["."],
    visibility = ["//py:__pkg__"],
    deps = [
        requirement("sympy"),
        requirement("matplotlib"),
    ],
)

py_library(
    name = "microbenchmark",
    testonly = True,
    srcs = ["runtime_py/microbenchmark.py"],
    imports = ["."],
    visibility = ["//visibility:public"],
)

py_library(
    name = "sklearn",
    srcs = ["runtime_py/sklearn.py"],
    imports = ["."],
    visibility = [
        "//problemdefinition:__pkg__",
        "//py:__pkg__",
    ],
    deps = [
        "//backend_py:compile-ekf",
        requirement("scikit-learn"),
    ],
)

py_test_suite(
    name = "scikit-learn-test",
    srcs = glob(
        [
            "test/scikit_learn_*.py",
        ],
        allow_empty = False,
    ),
    deps = [
        ":sklearn",
        "//problemdefinition:model",
    ] + PY_TEST_DEPS,
)
