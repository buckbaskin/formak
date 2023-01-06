load("@rules_python//python:defs.bzl", "py_binary")
load("@pip_deps//:requirements.bzl", "requirement")

cc_library(
    name = "hello-library",
    srcs = ["src/hello_greet.cpp"],
    hdrs = ["include/experimental/hello_greet.h"],
    strip_include_prefix = "include",
)

cc_binary(
    name = "hello-world",
    srcs = ["src/hello_world.cpp"],
    deps = [
        ":hello-library",
    ],
)

cc_test(
    name = "hello-test",
    srcs = ["test/hello_test.cpp"],
    deps = [
        ":hello-library",
        "@gtest//:gtest_main",
    ],
)

py_library(
    name = "hello-library-py",
    srcs = ["src/hello_greet.py"],
    imports = ["src"],
)

py_binary(
    name = "hello-world-py",
    srcs = ["src/hello_world.py"],
    main = "src/hello_world.py",
    deps = [":hello-library-py"],
)

### Start cpp-gen
py_binary(
    name = "cpp-gen-impl",
    srcs = ["src/cpp_gen.py"],
    main = "src/cpp_gen.py",
    deps = [
        requirement("sympy"),
        requirement("Jinja2"),
    ],
)

genrule(
    name = "cpp-gen",
    srcs = [":cpp-gen-impl"] + glob(["templates/*.cpp"]),
    outs = ["generated/jinja_basic_class.h"],
    cmd = "python3 $(location src/cpp_gen.py) --headertemplate $(location templates/basic_class.h) --sourcetemplate $(location templates/basic_class.cpp) --header $(location generated/jinja_basic_class.h) --source $(location generated/jinja_basic_class.h)",
    tools = ["src/cpp_gen.py"],
)

cc_library(
    name = "cpp-gen-cpp",
    hdrs = ["generated/jinja_basic_class.h"],
    strip_include_prefix = "generated",
    # deps = [":py-to-cpp-codegen"],
)

cc_test(
    name = "cpp-gen-cpp-test",
    srcs = ["test/cpp_gen_test.cpp"],
    deps = [
        ":cpp-gen-cpp",
        "@gtest//:gtest_main",
    ],
)
### End cpp-gen

py_test(
    name = "hello-test-py",
    srcs = ["test/hello_test.py"],
    main = "test/hello_test.py",
    deps = [
        ":hello-library-py",
        requirement("pytest"),
    ],
)

py_binary(
    name = "codegen-impl",
    srcs = ["src/py_to_cpp_codegen.py"],
    imports = ["src"],
    main = "src/py_to_cpp_codegen.py",
)

genrule(
    name = "py-to-cpp-codegen",
    srcs = ["src/py_to_cpp_codegen.py"],
    outs = [
        "generated/include/codegen/codegen.h",
        "generated/src/codegen.cpp",
    ],
    cmd = "python3 $(location src/py_to_cpp_codegen.py) --header $(location generated/include/codegen/codegen.h) --source $(location generated/src/codegen.cpp)",
)

py_test(
    name = "py-to-cpp-codegen-test",
    srcs = ["test/py_to_cpp_codegen_test.py"],
    main = "test/py_to_cpp_codegen_test.py",
    deps = [
        ":codegen-impl",
        requirement("pytest"),
    ],
)

cc_library(
    name = "codegen-output",
    srcs = ["generated/src/codegen.cpp"],
    hdrs = ["generated/include/codegen/codegen.h"],
    strip_include_prefix = "generated/include",
    # deps = [":py-to-cpp-codegen"],
)

cc_test(
    name = "codegen-output-test",
    srcs = ["test/codegen_output_test.cpp"],
    deps = [
        ":codegen-output",
        "@gtest//:gtest_main",
    ],
)

cc_test(
    name = "rapidcheckcheck",
    srcs = ["test/rapidcheckcheck.cpp"],
    deps = ["@rapidcheck"],
)

py_test(
    name = "hypothesischeck",
    srcs = ["test/hypothesischeck_test.py"],
    main = "test/hypothesischeck_test.py",
    deps = [
        requirement("hypothesis"),
        requirement("pytest"),
    ],
)

py_binary(
    name = "graph-editing",
    srcs = ["src/graph_editing.py"],
    main = "src/graph_editing.py",
    deps = [
        requirement("sympy"),
    ],
)

py_binary(
    name = "extract-graph-patterns",
    srcs = ["src/extract_graph_patterns.py"],
    main = "src/extract_graph_patterns.py",
    deps = [
        requirement("sympy"),
    ],
)

py_binary(
    name = "coffman-graham",
    srcs = ["src/coffman_graham.py"],
    main = "src/coffman_graham.py",
    deps = [
        requirement("sympy"),
    ],
)

py_binary(
    name = "linear-cpu-model",
    srcs = ["src/linear_cpu_model.py"],
    main = "src/linear_cpu_model.py",
)
