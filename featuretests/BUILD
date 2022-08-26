load("@rules_python//python:defs.bzl", "py_binary")
load("@pip_deps//:requirements.bzl", "requirement")

# py_library(
#     name = "hello-library-py",
#     srcs = ["src/hello_greet.py"],
#     imports = ["src"],
# )
#
# py_binary(
#     name = "hello-world-py",
#     srcs = ["src/hello_world.py"],
#     main = "src/hello_world.py",
#     deps = [":hello-library-py"],
# )
#
# py_test(
#     name = "hello-test-py",
#     srcs = ["test/hello_test.py"],
#     main = "test/hello_test.py",
#     deps = [
#         ":hello-library-py",
#         requirement("pytest"),
#     ],
# )

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
