load("@pip_deps//:requirements.bzl", "requirement")
load("//py:defs.bzl", "cc_formak_model", "cc_test_suite", "py_test_suite")
load("@bazel_skylib//rules:run_binary.bzl", "run_binary")
load("//featuretests:test_deps.bzl", "CC_FEATURE_TEST_DEPS", "PY_FEATURE_TEST_DEPS")

py_library(
    name = "model",
    srcs = ["model.py"],
    deps = ["//py:formak"],
)

py_library(
    name = "data",
    srcs = ["data.py"],
    deps = [
        "//py:formak",
        requirement("numpy"),
    ],
)

py_test_suite(
    name = "synthetic-data-test",
    srcs = ["synthetic_data_test.py"],
    deps = ["//py:formak", ":model", ":data"] + PY_FEATURE_TEST_DEPS,
)
