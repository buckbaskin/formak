load("@rules_python//python:defs.bzl", "py_binary")
load("@pip_deps//:requirements.bzl", "requirement")

py_binary(
    name = "symbolic-model",
    srcs = ["src/symbolic_model.py"],
    main = "src/symbolic_model.py",
    deps = [
        "//py:formak",
    ],
)

py_binary(
    name = "ekf",
    srcs = ["src/ekf.py"],
    main = "src/ekf.py",
    deps = [
        "//py:formak",
    ],
)
