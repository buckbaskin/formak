load("//py:defs.bzl", "cc_formak_model", "cc_test_suite")
load("//featuretests:test_deps.bzl", "CC_FEATURE_TEST_DEPS")

cc_formak_model(
    name = "cpp-model",
    namespace = "formak",
    pydeps = [],
    pymain = "generator.py",
    pysrcs = glob(
        ["*.py"],
        allow_empty = False,
    ),
)

cc_formak_model(
    name = "cpp-ekf",
    namespace = "formak",
    pydeps = [],
    pymain = "generator_ekf.py",
    pysrcs = glob(
        ["*.py"],
        allow_empty = False,
    ),
)

cc_test_suite(
    name = "cpp-library-for-model-evaluation",
    srcs = glob(
        ["*.cpp"],
        allow_empty = False,
    ),
    deps = [
        ":cpp-model",
        ":cpp-ekf",
    ] + CC_FEATURE_TEST_DEPS,
)
