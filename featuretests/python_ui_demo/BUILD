load("//py:defs.bzl", "py_test_suite")
load("//featuretests:test_deps.bzl", "PY_FEATURE_TEST_DEPS")

py_test_suite(
    name = "python-ui-demo",
    size = "small",
    srcs = [
        "orbital_test.py",
        "simple_test.py",
    ],
    deps = [
        "//formak:formak",
    ] + PY_FEATURE_TEST_DEPS,
)
