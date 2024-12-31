package(default_visibility = ["//visibility:private"])

load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "model",
    srcs = ["formak/problemdefinition/model.py"],
    imports = ["."],
    visibility = ["//py:__pkg__"],
    deps = [
        requirement("sympy"),
    ],
)
