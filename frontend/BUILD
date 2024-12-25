package(default_visibility = ["//compiler:__pkg__"])

load("@pip_deps//:requirements.bzl", "requirement")

py_library(
    name = "ui-model-base",
    srcs = ["formak/ui_model_base.py"],
    imports = ["."],
    visibility = ["//visibility:private"],
    deps = [
        requirement("sympy"),
        "//py:exceptions",
    ],
)

py_library(
    name = "ui-model",
    srcs = ["formak/ui_model.py"],
    imports = ["."],
    visibility = ["//visibility:private"],
    deps = [
        requirement("sympy"),
        "//py:exceptions",
        ":ui-model-base",
    ],
)

py_library(
    name = "ui-state-machine",
    srcs = ["formak/ui_state_machine.py"],
    imports = ["."],
    visibility = ["//visibility:private"],
    deps = [
        requirement("scikit-learn"), # this seems like an odd dependency, maybe some indirection could help here
    ],
)

py_library(
    name = "ui",
    srcs = ["formak/ui.py"],
    imports = ["."],
    deps = [
        ":ui-model",
        ":ui-state-machine",
    ],
    visibility = ["//visibility:public"],
)

