package(default_visibility = ["//visibility:private"])

py_library(
    name = "config",
    srcs = ["formak/compiler/config.py"],
    imports = ["."],
    visibility = ["//visibility:public"],
)

py_library(
    name = "basic-block",
    srcs = ["formak/compiler/basic_block.py"],
    imports = ["."],
    visibility = ["//visibility:public"],
)

py_library(
    name = "lib-compiler",
    srcs = ["formak/compiler/__init__.py"],
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = ["//formak:ui-model-base"],
)
