load("@pip_deps//:requirements.bzl", "requirement")
load("//py:defs.bzl", "py_test_suite", "cc_test_suite")

FEATURE_TEST_DEPS = [
    requirement("pytest"),
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
    ] + FEATURE_TEST_DEPS,
)

py_test_suite(
    name = "python-library-for-model-evaluation",
    srcs = glob(
        ["python_library_for_model_evaluation/*.py"],
        allow_empty = False,
    ),
    deps = [
        "//py:formak",
    ] + FEATURE_TEST_DEPS,
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
    ] + FEATURE_TEST_DEPS,
)

cc_test_suite(
    name = "cpp-library-for-model-evaluation",
    srcs = glob(
        ["cpp_library_for_model_evaluation/*.cpp"],
        allow_empty = False,
    ),
    deps = [
        "//py:formak",
    ] + FEATURE_TEST_DEPS,
)
