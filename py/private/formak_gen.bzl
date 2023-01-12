load("@pip_deps//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_binary", "py_library")

def cc_formak_model(name, pymain, pysrcs, pydeps = None, python_version = None, imports = None, visibility = None, **kwargs):
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
    OUTPUT_HEADER = "generated/jinja_basic_class.h"
    OUTPUT_SOURCE = "generated/jinja_basic_class.cpp"
    OUTPUT_FILES = [
        OUTPUT_HEADER,
        OUTPUT_SOURCE,
    ]

    # TODO(buck): Parameterize the output command
    native.genrule(
        name = GENRULE_NAME,
        srcs = [PY_BINARY_NAME, "//py:templates"],
        outs = OUTPUT_FILES,
        cmd = "python3 $(location " + pymain + ") --templates $(locations " + MODEL_TEMPLATES + ") --header $(location generated/jinja_basic_class.h) --source $(location generated/jinja_basic_class.cpp)",
        tools = [pymain],
        visibility = ["//visibility:private"],
    )

    native.cc_library(
        name = CC_LIBRARY_NAME,
        srcs = [OUTPUT_SOURCE],
        hdrs = [OUTPUT_HEADER],
        strip_include_prefix = "generated",
        deps = [],
        visibility = visibility,
    )
