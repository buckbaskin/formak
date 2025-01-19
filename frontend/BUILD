package(default_visibility = ["//visibility:private"])

load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "model-validation",
    srcs = ["frontend/model_validation.py"],
    imports = ["."],
    visibility = [
        "//backend_cpp:__pkg__",
        "//backend_py:__pkg__",
        "//py:__pkg__",
    ],
)
