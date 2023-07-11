load("@pip_deps//:requirements.bzl", "requirement")
load("//py:defs.bzl", "py_test_suite")
load("//featuretests:test_deps.bzl", "PY_FEATURE_TEST_DEPS")

py_test_suite(
    name = "scikit-learn-integration",
    srcs = [
        "pipeline_test.py",
        "simple_regression_test.py",
        "ui_test.py",
    ],
    deps = [
        "//py:formak",
        requirement("numpy"),
    ] + PY_FEATURE_TEST_DEPS,
)
