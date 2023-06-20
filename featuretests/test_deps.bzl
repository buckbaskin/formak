load("@pip_deps//:requirements.bzl", "requirement")

PY_FEATURE_TEST_DEPS = [
    requirement("pytest"),
]

CC_FEATURE_TEST_DEPS = [
    "@gtest//:gtest_main",
]
