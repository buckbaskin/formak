load("@rules_python//python:defs.bzl", "py_binary")
load("@pip_deps//:requirements.bzl", "requirement")

FEATURE_TEST_DEPS = [
    requirement("pytest"),
]

py_test(
    name = "python-ui-demo-simple",
    srcs = ["python_ui_demo/simple_test.py"],
    main = "python_ui_demo/simple_test.py",
    deps = [
        "//py:formak",
    ] + FEATURE_TEST_DEPS,
)

py_test(
    name = "python-ui-demo-orbital",
    srcs = ["python_ui_demo/orbital_test.py"],
    main = "python_ui_demo/orbital_test.py",
    deps = [
        "//py:formak",
    ] + FEATURE_TEST_DEPS,
)

py_test(
    name = "python-library-for-model-evaluation-compilation-performance-test",
    srcs = ["python_library_for_model_evaluation/compilation_performance_test.py"],
    main = "python_library_for_model_evaluation/compilation_performance_test.py",
    deps = [
        "//py:formak",
    ] + FEATURE_TEST_DEPS,
)

py_test(
    name = "python-library-for-model-evaluation-compilation",
    srcs = ["python_library_for_model_evaluation/simple_compilation_test.py"],
    main = "python_library_for_model_evaluation/simple_compilation_test.py",
    deps = [
        "//py:formak",
    ] + FEATURE_TEST_DEPS,
)

py_test(
    name = "python-library-for-model-evaluation-simple",
    srcs = ["python_library_for_model_evaluation/simple_test.py"],
    main = "python_library_for_model_evaluation/simple_test.py",
    deps = [
        "//py:formak",
    ] + FEATURE_TEST_DEPS,
)

py_test(
    name = "python-library-for-model-evaluation-ekf",
    srcs = ["python_library_for_model_evaluation/simple_to_ekf_test.py"],
    main = "python_library_for_model_evaluation/simple_to_ekf_test.py",
    deps = [
        "//py:formak",
    ] + FEATURE_TEST_DEPS,
)
