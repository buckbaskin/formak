load("@pip_deps//:requirements.bzl", "requirement")
load("//py:defs.bzl", "py_test_suite")

FEATURE_TEST_DEPS = [
    requirement("pytest"),
]

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
