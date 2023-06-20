load("@pip_deps//:requirements.bzl", "requirement")
load("//py:defs.bzl", "cc_formak_model", "cc_test_suite", "py_test_suite")
load("//featuretests:test_deps.bzl", "CC_FEATURE_TEST_DEPS", "PY_FEATURE_TEST_DEPS")

cc_formak_model(
    name = "cpp-rocket-model",
    namespace = "featuretest",
    pydeps = [],
    pymain = "generator.py",
    pysrcs = glob(
        ["*.py"],
        allow_empty = False,
    ),
)

py_test_suite(
    name = "rocket-model-py-test",
    srcs = glob(
        ["*.py"],
        allow_empty = False,
    ),
    deps = [
        "//py:formak",
        requirement("numpy"),
    ] + PY_FEATURE_TEST_DEPS,
)

cc_test_suite(
    name = "rocket-model-cpp-test",
    srcs = glob(
        ["*.cpp"],
        allow_empty = False,
    ),
    deps = [
        ":cpp-rocket-model",
    ] + CC_FEATURE_TEST_DEPS,
)
