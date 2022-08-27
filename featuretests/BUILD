load("@rules_python//python:defs.bzl", "py_binary")
load("@pip_deps//:requirements.bzl", "requirement")

py_test(
    name = "python-ui-demo-simple",
    srcs = ["test/python_ui_demo_simple_test.py"],
    main = "test/python_ui_demo_simple_test.py",
    deps = [
        requirement("pytest"),
        "//formak:ui",
    ],
)

py_test(
    name = "python-ui-demo-orbital",
    srcs = ["test/python_ui_demo_orbital_test.py"],
    main = "test/python_ui_demo_orbital_test.py",
    deps = [
        requirement("pytest"),
        "//formak:ui",
    ],
)
