load("@rules_python//python:defs.bzl", "py_binary")
load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "ui",
    srcs = [
        "src/formak/__init__.py",
        "src/formak/ui.py",
    ],
    imports = ["src"],
    visibility = ["//visibility:public"],
    deps = [
        requirement("sympy"),
    ],
)

py_test(
    name = "ui-test-py",
    srcs = ["test/ui_test.py"],
    main = "test/ui_test.py",
    deps = [
        ":ui",
        requirement("pytest"),
    ],
)
