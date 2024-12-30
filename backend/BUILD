package(default_visibility = ["//compiler:__pkg__"])

load("//py:defs.bzl", "PY_TEST_DEPS", "py_test_suite")
load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "backend",
    srcs = [],
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [
        ":ast-fragments",
        ":ast-tools",
    ],
)

py_library(
    name = "ast-tools",
    srcs = ["formak/backend/ast_tools.py"],
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [
        requirement("Jinja2"),
    ],
)

py_library(
    name = "ast-fragments",
    srcs = ["formak/backend/ast_fragments.py"],
    imports = ["."],
    deps = [
        ":ast-tools",
    ],
)
