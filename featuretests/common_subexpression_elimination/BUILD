load("@pip_deps//:requirements.bzl", "requirement")
load("//py:defs.bzl", "cc_formak_model", "cc_test_suite", "py_test_suite")
load("//featuretests:test_deps.bzl", "CC_FEATURE_TEST_DEPS", "PY_FEATURE_TEST_DEPS")

py_library(
    name = "common",
    srcs = ["common.py"],
    deps = ["//py:formak"],
)

py_test_suite(
    name = "py-test",
    srcs = ["simple_test.py"],
    deps = [
        "//py:formak",
        ":common",
        requirement("numpy"),
    ] + PY_FEATURE_TEST_DEPS,
)

cc_formak_model(
    name = "cse-model",
    namespace = "cse",
    pydeps = [
        ":common",
    ],
    pymain = "gen_cse.py",
    pysrcs = glob(
        ["*.py"],
        allow_empty = False,
    ),
)

cc_formak_model(
    name = "no-cse-model",
    namespace = "no_cse",
    pydeps = [
        ":common",
    ],
    pymain = "gen_no_cse.py",
    pysrcs = glob(
        ["*.py"],
        allow_empty = False,
    ),
)

cc_test(
    name = "cpp-test",
    srcs = ["simple_test.cpp"],
    deps = [
        ":cse-model",
        ":no-cse-model",
        "//cpp:utils-microbenchmark",
    ] + CC_FEATURE_TEST_DEPS,
)
