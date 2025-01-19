package(default_visibility = ["//visibility:private"])

load("@pip_deps//:requirements.bzl", "requirement")
load("//py:defs.bzl", "PY_TEST_DEPS", "py_test_suite")

py_library(
    name = "ui-model-base",
    srcs = ["problemdefinition/ui_model_base.py"],
    imports = ["."],
    visibility = [
        "//backend_py:__pkg__",
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
        ":config-view",
        ":model",
        ":nis-score",
        "//backend_py:sklearn",
    ],
)

py_library(
    name = "config-view",
    srcs = ["problemdefinition/config_view.py"],
    imports = ["."],
    visibility = ["//py:__pkg__"],
)

py_library(
    name = "nis-score",
    srcs = ["problemdefinition/nis_score.py"],
    imports = ["."],
    visibility = ["//py:__pkg__"],
)

py_test_suite(
    name = "ui-test",
    srcs = [
        "test/ui_test.py",
    ],
    deps = [
        ":model",
        ":config-view",
        ":nis-score",
        "//backend_py:sklearn",
    ] + PY_TEST_DEPS,
)

py_test_suite(
    name = "design-manager-test",
    srcs = [
        "test/design_manager_test.py",
    ],
    deps = [
        ":design-manager",
    ] + PY_TEST_DEPS,
)
