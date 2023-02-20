load("@pip_deps//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load("@bazel_skylib//rules:run_binary.bzl", "run_binary")

def cc_formak_model(namespace, name, pymain, pysrcs, pydeps = None, python_version = None, imports = None, visibility = None, **kwargs):
    PY_LIBRARY_NAME = name + "py-library-formak-model"
    PY_BINARY_NAME = name + "py-binary-formak-model"
    GENRULE_NAME = name + "genrule-formak-model"
    CC_LIBRARY_NAME = name

    ALWAYS_PY_DEPS = [
        "//py:formak",
        requirement("sympy"),
        requirement("Jinja2"),
    ]

    if pydeps == None:
        pydeps = []

    py_library(
        name = PY_LIBRARY_NAME,
        srcs = pysrcs,
        deps = pydeps + ALWAYS_PY_DEPS,
        imports = imports,
        visibility = ["//visibility:private"],
    )

    py_binary(
        name = PY_BINARY_NAME,
        srcs = [pymain],
        main = pymain,
        deps = [PY_LIBRARY_NAME],
        visibility = ["//visibility:private"],
    )

    MODEL_TEMPLATES = "//py:templates"

    # TODO(buck): Use name to give these better names, maybe namespace too
    OUTPUT_HEADER = "generated/%s/%s.h" % (namespace, name)
    OUTPUT_SOURCE = "generated/%s/%s.cpp" % (namespace, name)
    OUTPUT_FILES = [
        OUTPUT_HEADER,
        OUTPUT_SOURCE,
    ]

    # TODO(buck): Parameterize the output command
    run_binary(
        name = GENRULE_NAME,
        tool = PY_BINARY_NAME,
        args = ["--templates", "$(locations " + MODEL_TEMPLATES + ")", "--header", "$(location generated/%s/%s.h)" % (namespace, name), "--source", "$(location generated/%s/%s.cpp)" % (namespace, name), "--namespace", namespace],
        outs = OUTPUT_FILES,
        srcs = ["//py:templates"],
    )

    native.cc_library(
        name = CC_LIBRARY_NAME,
        srcs = [OUTPUT_SOURCE],
        hdrs = [OUTPUT_HEADER],
        strip_include_prefix = "generated",
        deps = ["@eigen//:eigen"],
        visibility = visibility,
    )
