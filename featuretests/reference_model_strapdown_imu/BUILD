load("@pip_deps//:requirements.bzl", "requirement")
load("//py:defs.bzl", "cc_formak_model", "cc_test_suite", "py_test_suite")
load("@bazel_skylib//rules:run_binary.bzl", "run_binary")
load("//featuretests:test_deps.bzl", "CC_FEATURE_TEST_DEPS", "PY_FEATURE_TEST_DEPS")

py_test_suite(
    name = "small-test",
    srcs = ["small_test.py"],
    data = [
        "NASA_sample.csv",
    ],
    deps = ["//py:formak"] + PY_FEATURE_TEST_DEPS,
)

py_test_suite(
    name = "preignition-test",
    srcs = ["preignition_test.py"],
    data = [
        "NASA_preignition.csv",
    ],
    deps = ["//py:formak"] + PY_FEATURE_TEST_DEPS,
)
