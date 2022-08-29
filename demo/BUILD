load("@rules_python//python:defs.bzl", "py_binary")
load("@pip_deps//:requirements.bzl", "requirement")

py_binary(
    name = "orbital-model",
    srcs = ["src/orbital_model.py"],
    main = "src/orbital_model.py",
    deps = [
        requirement("sympy"),
        "//formak:user-interface",
    ],
)
