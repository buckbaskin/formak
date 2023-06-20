load("@pip_deps//:requirements.bzl", "requirement")
load("//py:defs.bzl", "cc_formak_model", "cc_test_suite", "py_test_suite")
load("@bazel_skylib//rules:run_binary.bzl", "run_binary")
load("//featuretests:test_deps.bzl", "CC_FEATURE_TEST_DEPS", "PY_FEATURE_TEST_DEPS")

cc_formak_model(
    name = "cpp-model",
    namespace = "formak",
    pydeps = [],
    pymain = "cpp_library_for_model_evaluation/generator.py",
    pysrcs = glob(
        ["cpp_library_for_model_evaluation/*.py"],
        allow_empty = False,
    ),
)

cc_formak_model(
    name = "cpp-ekf",
    namespace = "formak",
    pydeps = [],
    pymain = "cpp_library_for_model_evaluation/generator_ekf.py",
    pysrcs = glob(
        ["cpp_library_for_model_evaluation/*.py"],
        allow_empty = False,
    ),
)

cc_formak_model(
    name = "cpp-rocket-model",
    namespace = "featuretest",
    pydeps = [],
    pymain = "rocket_model/generator.py",
    pysrcs = glob(
        ["rocket_model/*.py"],
        allow_empty = False,
    ),
)

cc_test_suite(
    name = "cpp-library-for-model-evaluation",
    srcs = glob(
        ["cpp_library_for_model_evaluation/*.cpp"],
        allow_empty = False,
    ),
    deps = [
        ":cpp-model",
        ":cpp-ekf",
    ] + CC_FEATURE_TEST_DEPS,
)

py_test_suite(
    name = "rocket-model-py-test",
    srcs = glob(
        ["rocket_model/*.py"],
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
        ["rocket_model/*.cpp"],
        allow_empty = False,
    ),
    deps = [
        ":cpp-rocket-model",
    ] + CC_FEATURE_TEST_DEPS,
)

py_binary(
    name = "ast-generate-cpp-tool",
    srcs = ["ast_code_generation/generator.py"],
    main = "ast_code_generation/generator.py",
    deps = ["//py:formak"],
)

run_binary(
    name = "ast-generate-cpp",
    srcs = ["//py:templates"],
    outs = [
        "generated/example.cpp",
        "generated/example.h",
    ],
    args = [
        "$(location generated/example.h)",
        "$(location generated/example.cpp)",
    ],
    tool = "ast-generate-cpp-tool",
)

cc_library(
    name = "ast-cpp-generated",
    srcs = ["generated/example.cpp"],
    hdrs = ["generated/example.h"],
    strip_include_prefix = "generated",
    deps = ["@eigen"],
)

cc_test_suite(
    name = "ast-code-generation-cpp-test",
    srcs = glob(
        ["ast_code_generation/*.cpp"],
        allow_empty = False,
    ),
    deps = [
        ":ast-cpp-generated",
    ] + CC_FEATURE_TEST_DEPS,
)

py_test_suite(
    name = "cse-py-test",
    srcs = glob(
        ["common_subexpression_elimination/*.py"],
        allow_empty = False,
    ),
    deps = [
        "//py:formak",
        requirement("numpy"),
    ] + PY_FEATURE_TEST_DEPS,
)
