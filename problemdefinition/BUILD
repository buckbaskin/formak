package(default_visibility = ["//visibility:private"])

load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "model",
    srcs = ["problemdefinition/model.py"],
    imports = ["."],
    visibility = [
        "//py:__pkg__",
        "//runtime_py:__pkg__",
    ],
    deps = [
        requirement("sympy"),
    ],
)

py_library(
    name = "design-manager",
    srcs = ["problemdefinition/design_manager.py"],
    imports = ["."],
    visibility = ["//py:__pkg__"],
    deps = [
        requirement("scikit-learn"),
        "//py:python",
    ],
)
