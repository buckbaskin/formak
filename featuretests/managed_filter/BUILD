load("@pip_deps//:requirements.bzl", "requirement")
load("//py:defs.bzl", "cc_formak_model", "cc_test_suite", "py_test_suite")
load("@bazel_skylib//rules:run_binary.bzl", "run_binary")
load("//featuretests:test_deps.bzl", "CC_FEATURE_TEST_DEPS", "PY_FEATURE_TEST_DEPS")

py_test_suite(
    name = "tick-interface-py",
    srcs = ["tick_interface_test.py"],
    deps = ["//py:formak"] + PY_FEATURE_TEST_DEPS,
)

cc_formak_model(
    name = "example",
    namespace = "featuretest",
    pydeps = [],
    pymain = "generator.py",
    pysrcs = glob(
        ["*.py"],
        allow_empty = False,
    ),
)

cc_test(
    name = "tick-interface-cpp",
    srcs = ["tick_interface_test.cpp"],
    deps =
        [
            "//cpp/runtime:managed-filter",
            ":example",
        ] + CC_FEATURE_TEST_DEPS,
)

cc_test(
    name = "tick-overhead-test-cpp",
    srcs = ["tick_overhead_test.cpp"],
    deps =
        [
            "//cpp/runtime:managed-filter",
        "//cpp:utils-microbenchmark",
            ":example",
        ] + CC_FEATURE_TEST_DEPS,
)
