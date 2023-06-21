load("@pip_deps//:requirements.bzl", "requirement")
load("//py:defs.bzl", "py_test_suite")
load("//featuretests:test_deps.bzl", "PY_FEATURE_TEST_DEPS")

py_test_suite(
    name = "py-test",
    srcs = ["simple_test.py"],
    deps = [
        "//py:formak",
        requirement("numpy"),
    ] + PY_FEATURE_TEST_DEPS,
)
