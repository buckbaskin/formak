load("@pip_deps//:requirements.bzl", "requirement")
load("//py:defs.bzl", "cc_formak_model", "cc_test_suite", "py_test_suite")

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
    name = "rocket-model",
    srcs = glob(
        ["rocket_model/*.py"],
        allow_empty = False,
    ),
    deps = [
        "//py:formak",
        requirement("numpy"),
    ] + PY_FEATURE_TEST_DEPS,
)
