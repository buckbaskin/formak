load("//py:defs.bzl", "py_test_suite")
load("//featuretests:test_deps.bzl", "PY_FEATURE_TEST_DEPS")

py_test_suite(
    name = "python-model",
    srcs = [
        "simple_test.py",
    ],
    deps = [
        "//py:formak",
    ] + PY_FEATURE_TEST_DEPS,
)

py_test(
    name = "python-ekf",
    srcs = ["simple_to_ekf_test.py"],
    main = "simple_to_ekf_test.py",
    deps = [
        "//py:formak",
    ] + PY_FEATURE_TEST_DEPS,
)
