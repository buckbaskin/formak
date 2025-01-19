package(default_visibility = ["//visibility:private"])

load("@pip_deps//:requirements.bzl", "requirement")
load("//py:defs.bzl", "PY_TEST_DEPS", "py_test_suite")

# TODO: this should be backend?
py_library(
    name = "model-validation",
    srcs = ["frontend/model_validation.py"],
    imports = ["."],
    visibility = [
        "//backend_cpp:__pkg__",
        "//backend_py:__pkg__",
        "//py:__pkg__",
    ],
)

py_test_suite(
    name = "rotation-symbolic-test",
    srcs = [
        "test/rotation_symbolic_test.py",
    ],
    deps = [
        requirement("sympy"),
    ] + PY_TEST_DEPS,
)
