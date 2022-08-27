load("@rules_python//python:defs.bzl", "py_binary")
load("@pip_deps//:requirements.bzl", "requirement")

py_test(
    name = "python-ui-demo-simple",
    srcs = ["python_ui_demo/simple_test.py"],
    main =  "python_ui_demo/simple_test.py",
    deps = [
        requirement("pytest"),
        "//formak:ui",
    ],
)

py_test(
    name = "python-ui-demo-orbital",
    srcs = ["python_ui_demo/orbital_test.py"],
    main =  "python_ui_demo/orbital_test.py",
    deps = [
        requirement("pytest"),
        "//formak:ui",
    ],
)
