package(default_visibility = ["//visibility:private"])

load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "ui-model-base",
    srcs = ["problemdefinition/ui_model_base.py"],
    imports = ["."],
    visibility = [
        "//py:__pkg__",
    ],
)

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
        ":ui-model-base",
    ],
)

# Allow Model for use in testing in other packages
py_library(
    name = "model-for-testing",
    testonly = True,
    imports = ["."],
    visibility = [
        "//backend_cpp:__pkg__",
    ],
    deps = [
        ":model",
    ],
)

py_library(
    name = "design-manager",
    srcs = ["problemdefinition/design_manager.py"],
    imports = ["."],
    visibility = ["//py:__pkg__"],
    deps = [
        requirement("scikit-learn"),
        "//backend_py:sklearn",
        "//py:python",
    ],
)
