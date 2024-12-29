package(default_visibility = ["//compiler:__pkg__"])

load("//py:defs.bzl", "PY_TEST_DEPS", "py_test_suite")
load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "backend",
    srcs = ['formak/backend/__init__.py'],
    imports = ["."],
    deps = [
        ":ast-tools",
        ":ast-fragments",
    ],
    visibility = ["//formak:__pkg__"],
)

py_library(
    name = "ast-tools",
    srcs = ["formak/backend/ast_tools.py"],
    imports = ["."],
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

