package(default_visibility = ["//visibility:private"])

load("@pip_deps//:requirements.bzl", "requirement")
load("//py:defs.bzl", "PY_TEST_DEPS", "py_test_suite")

filegroup(
    name = "templates",
    srcs = glob(
        [
            "formak/templates/*.hpp",
            "formak/templates/*.cpp",
            "formak/templates/test/*.cpp",
        ],
        allow_empty = False,
    ),
    visibility = ["//visibility:public"],
)

py_library(
    name = "ast",
    srcs = ["backend_cpp/ast.py"],
    imports = ["."],
    visibility = [
        "//featuretests/ast_code_generation:__pkg__",
        "//py:__pkg__",
    ],
    deps = [
        requirement("Jinja2"),
    ],
)

py_test_suite(
    name = "ast-test",
    srcs = [
        "test/ast_test.py",
    ],
    deps = [
        ":ast",
    ] + PY_TEST_DEPS,
)

py_library(
    name = "ast-fragments",
    srcs = ["backend_cpp/ast_fragments.py"],
    imports = ["."],
    visibility = [
        "//py:__pkg__",
    ],
    deps = [
        ":ast",
    ],
)
