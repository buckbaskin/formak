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

py_binary(
    name = "example",
    srcs = ["cpp_library_for_model_evaluation/generator.py"],
    main = "cpp_library_for_model_evaluation/generator.py",
    deps = [
        "//py:formak",
        requirement("Jinja2"),
    ] + PY_FEATURE_TEST_DEPS,
)

cc_formak_model(
    name = "cpp-model",
    pydeps = [],
    pymain = "cpp_library_for_model_evaluation/generator.py",
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
    ] + CC_FEATURE_TEST_DEPS,
)
