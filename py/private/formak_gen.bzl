load("@pip_deps//:requirements.bzl", "requirement")
load("@rules_python//python:defs.bzl", "py_binary")

def cc_formak_model(name, pymain, pysrcs, pydeps = None, python_version = None, ccdeps = None, visibility = None, **kwargs):
    PY_BINARY_NAME = name + "py-binary-formak-model"
    GENRULE_NAME = name + "genrule-formak-model"
    CC_LIBRARY_NAME = name

    ALWAYS_PY_DEPS = [
        requirement("sympy"),
        requirement("Jinja2"),
    ]

    if pydeps == None:
        pydeps = []

    py_binary(
        name = PY_BINARY_NAME,
        srcs = pysrcs,
        main = pymain,
        deps = pydeps + ALWAYS_PY_DEPS,
        visibility = ["//visibility:private"],
    )

    MODEL_TEMPLATES_HEADER = "templates/formak_model.h"
    MODEL_TEMPLATES_SOURCE = "templates/formak_model.cpp"
    MODEL_TEMPLATES = [
        MODEL_TEMPLATES_SOURCE,
        MODEL_TEMPLATES_HEADER,
    ]

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
        cmd = "python3 $(location " + pymain + ") --headertemplate " + MODEL_TEMPLATES_HEADER + " --sourcetemplate " + MODEL_TEMPLATES_SOURCE + " --header $(location generated/jinja_basic_class.h) --source $(location generated/jinja_basic_class.cpp)",
        tools = [pymain],
        visibility = ["//visibility:private"],
    )

    native.cc_library(
        name = CC_LIBRARY_NAME,
        srcs = [OUTPUT_SOURCE],
        hdrs = [OUTPUT_HEADER],
        strip_include_prefix = "generated",
        deps = ccdeps,
        visibility = visibility,
    )
