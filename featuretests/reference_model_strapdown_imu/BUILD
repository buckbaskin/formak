load("@pip_deps//:requirements.bzl", "requirement")
load("//py:defs.bzl", "cc_formak_model", "cc_test_suite", "py_test_suite")
load("@bazel_skylib//rules:run_binary.bzl", "run_binary")
load("//featuretests:test_deps.bzl", "CC_FEATURE_TEST_DEPS", "PY_FEATURE_TEST_DEPS")

py_library(
    name = "model-definition",
    srcs = ["model_definition.py"],
    deps = ["//py:formak"],
)

py_binary(
    name = "model-definition-bin",
    srcs = ["model_definition.py"],
    main = "model_definition.py",
    deps = ["//py:formak"],
)
