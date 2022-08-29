load("@rules_python//python:defs.bzl", "py_binary")
load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "base",
    srcs = [
        "src/formak/__init__.py",
    ],
    imports = ["src"],
    visibility = ["//visibility:private"],
    deps = [],
)

py_library(
    name = "user-interface",
    srcs = [
        "src/formak/ui.py",
    ],
    imports = ["src"],
    visibility = ["//visibility:public"],
    deps = [
        requirement("sympy"),
    ],
)

py_library(
    name = "python",
    srcs = [
        "src/formak/python.py",
    ],
    imports = ["src"],
    visibility = ["//visibility:public"],
    deps = [],
)

py_test(
    name = "user-interface-test-py",
    srcs = ["test/ui_test.py"],
    main = "test/ui_test.py",
    deps = [
        ":user-interface",
        requirement("pytest"),
    ],
)

py_test(
    name = "py-test-py",
    srcs = ["test/python_test.py"],
    main = "test/python_test.py",
    deps = [
        ":python",
        ":user-interface",
        requirement("pytest"),
    ],
)
