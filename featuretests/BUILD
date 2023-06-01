load("@pip_deps//:requirements.bzl", "requirement")
load("//py:defs.bzl", "cc_formak_model", "cc_test_suite", "py_test_suite")
load("@bazel_skylib//rules:run_binary.bzl", "run_binary")

PY_FEATURE_TEST_DEPS = [
    requirement("pytest"),
]

CC_FEATURE_TEST_DEPS = [
    "@gtest//:gtest_main",
]

py_test_suite(
    name = "python-ui-demo",
    size = "small",
    srcs = glob(
        ["python_ui_demo/*.py"],
        allow_empty = False,
    ),
    deps = [
        "//py:formak",
    ] + PY_FEATURE_TEST_DEPS,
)

py_test_suite(
    name = "python-library-for-model-evaluation",
    srcs = glob(
        ["python_library_for_model_evaluation/*.py"],
        allow_empty = False,
    ),
    deps = [
        "//py:formak",
    ] + PY_FEATURE_TEST_DEPS,
)

py_test_suite(
    name = "scikit-learn-integration",
    srcs = glob(
        ["scikit_learn_integration/*.py"],
        allow_empty = False,
    ),
    deps = [
        "//py:formak",
        requirement("numpy"),
    ] + PY_FEATURE_TEST_DEPS,
)

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

py_test_suite(
    name = "ast-code-generation-py-test",
    srcs = glob(
        ["ast_code_generation/*.py"],
        allow_empty = False,
    ),
    deps = [
        "//py:formak",
    ] + PY_FEATURE_TEST_DEPS,
)

py_binary(
    name = "ast-generate-cpp-tool",
    srcs = ["ast_code_generation/generator.py"],
    main = "ast_code_generation/generator.py",
    deps = ["//py:formak"],
)

run_binary(
    name = "ast-generate-cpp",
    outs = ["generated/example.h"],
    args = ["$(location generated/example.h)"],
    tool = "ast-generate-cpp-tool",
)

cc_library(
    name = "ast-cpp-generated",
    srcs = [],
    hdrs = ["generated/example.h"],
    strip_include_prefix = "generated",
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
