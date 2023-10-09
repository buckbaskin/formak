## Start Selenium-based load
# https://github.com/SeleniumHQ/selenium/blob/c67d0f1dfc49c7f9f6938ac64ae676a348defebd/py/BUILD.bazel
load("@rules_python//python:defs.bzl", "py_library")
load("@rules_python//python:packaging.bzl", "py_package", "py_wheel")
load("@pip_deps//:requirements.bzl", "requirement")
load("//py:defs.bzl", "cc_formak_model", "py_test_suite")
load("@rules_python//python:pip.bzl", "compile_pip_requirements")
# TODOs
# - exporting as a pip package for non-bazel users
## End Selenium-based load

FORMAK_VERSION = "0.0.1-alpha00"

PY_TEST_DEPS = [
    requirement("hypothesis"),
    requirement("matplotlib"),
    requirement("PyQt5"),
    requirement("pytest"),
    requirement("pytest-mock"),
]

CC_TEST_DEPS = [
    "@gtest//:gtest_main",
]

filegroup(
    name = "templates",
    srcs = glob(
        [
            "formak/templates/*.hpp",
            "formak/templates/*.cpp",
            "formak/templates/test/*.cpp",
        ],
        allow_empty = False,
    ),
    visibility = ["//visibility:public"],
)

py_library(
    name = "ui",
    srcs = ["formak/ui.py"],
    imports = ["."],
    deps = [
        requirement("sympy"),
    ],
)

py_library(
    name = "python",
    srcs = ["formak/python.py"],
    imports = ["."],
    deps = [
        ":common",
        ":exceptions",
        requirement("sympy"),
        requirement("scipy"),
        requirement("numpy"),
    ],
)

py_library(
    name = "runtime",
    srcs = ["formak/runtime.py"],
    imports = ["."],
)

py_library(
    name = "microbenchmark",
    testonly = True,
    srcs = ["formak/microbenchmark.py"],
    imports = ["."],
    visibility = ["//visibility:public"],
)

py_library(
    name = "rotation",
    srcs = ["formak/rotation.py"],
    imports = ["."],
    deps = [
        requirement("sympy"),
    ],
)

py_library(
    name = "common",
    srcs = ["formak/common.py"],
    imports = ["."],
    deps = [
        ":exceptions",
        requirement("sympy"),
    ],
)

py_library(
    name = "exceptions",
    srcs = ["formak/exceptions.py"],
    imports = ["."],
    deps = [
    ],
)

py_library(
    name = "ast-tools",
    srcs = ["formak/ast_tools.py"],
    imports = ["."],
    deps = [
        requirement("Jinja2"),
    ],
)

py_library(
    name = "ast-fragments",
    srcs = ["formak/ast_fragments.py"],
    imports = ["."],
    deps = [
        ":ast-tools",
    ],
)

py_library(
    name = "cpp",
    srcs = ["formak/cpp.py"],
    imports = ["."],
    deps = [
        ":ast-fragments",
        ":ast-tools",
        ":common",
        ":exceptions",
        requirement("sympy"),
    ],
)

py_library(
    name = "reference-models",
    srcs = glob(
        ["formak/reference_models/*.py"],
        allow_empty = False,
    ),
    imports = ["."],
    deps = [
        ":python",
        ":ui",
        requirement("sympy"),
    ],
)

py_library(
    name = "formak",
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [
        ":cpp",
        ":python",
        ":reference-models",
        ":rotation",
        ":runtime",
        ":ui",
    ],
)

# TODO(buck): Write up the difference between py_library and py_package

py_package(
    name = "formak-pkg",
    packages = [
        "py.formak",
        "py.formak.cpp",
        "py.formak.python",
        "py.formak.ui",
    ],
    deps = [":formak"],
)

# This is borrowed from Selenium
#   - %s/selenium/formak/g
#   - Elected specific classifiers for Formak
#   - Added license, other metadata for Formak
py_wheel(
    name = "formak-wheel",
    classifiers = [
        "Development Status :: 1 - Planning",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Topic :: Software Development :: Libraries",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
    ],
    description_file = "README.md",
    distribution = "formak",
    homepage = "https://github.com/buckbaskin/formak",
    license = "MIT",
    python_requires = "~=3.8",
    python_tag = "py3",
    requires = [
        "sympy",
    ],
    strip_path_prefixes = [
        "py/",
    ],
    version = FORMAK_VERSION,
    visibility = ["//visibility:public"],
    deps = [
        ":formak-pkg",
    ],
)

py_library(
    name = "init-tree",
    testonly = True,
    srcs = [
        "conftest.py",
        "test/__init__.py",
        "test/formak/__init__.py",
    ],
    data = [
        "//common:pytest.ini",
        "//common:setup.cfg",
    ],
    imports = ["."],
    deps = [
    ],
)

py_test_suite(
    name = "unit-py",
    size = "small",
    srcs = glob(
        [
            "test/unit/*.py",
        ],
        allow_empty = False,
    ),
    deps = [
        ":init-tree",
        ":formak",
    ] + PY_TEST_DEPS,
)

py_test_suite(
    name = "cpp-generation-py",
    size = "small",
    srcs = glob(
        [
            "test/unit/cpp/*.py",
        ],
        allow_empty = False,
    ),
    deps = [
        ":init-tree",
        ":formak",
    ] + PY_TEST_DEPS,
)

py_test_suite(
    name = "python-generation-py",
    size = "small",
    srcs = glob(
        [
            "test/unit/python/*.py",
        ],
        allow_empty = False,
    ),
    deps = [
        ":init-tree",
        ":formak",
    ] + PY_TEST_DEPS,
)

py_test_suite(
    name = "runtime-test",
    size = "small",
    srcs = glob(
        [
            "test/unit/runtime/*.py",
        ],
        allow_empty = False,
    ),
    deps = [
        ":init-tree",
        ":formak",
    ] + PY_TEST_DEPS,
)

py_test_suite(
    name = "reference-models-test",
    size = "small",
    srcs = glob(
        [
            "test/unit/reference_models/*.py",
        ],
        allow_empty = False,
    ),
    deps = [
        ":init-tree",
        ":reference-models",
    ] + PY_TEST_DEPS,
)

py_test_suite(
    name = "integration-py",
    size = "medium",
    srcs = glob([
        "test/formak/**/*.py",
    ]),
    deps = [
        ":init-tree",
        ":formak",
    ] + PY_TEST_DEPS,
)

cc_formak_model(
    name = "simple-model",
    namespace = "unit",
    pydeps = [],
    pymain = "test/unit/cpp/generator.py",
    pysrcs = glob(
        ["test/unit/cpp/*.py"],
        allow_empty = False,
    ),
)

cc_formak_model(
    name = "simple-ekf",
    namespace = "unit",
    pydeps = [],
    pymain = "test/unit/cpp/generator_ekf.py",
    pysrcs = glob(
        ["test/unit/cpp/*.py"],
        allow_empty = False,
    ),
)

cc_formak_model(
    name = "ekf-cse",
    namespace = "cse",
    pydeps = [],
    pymain = "test/unit/cpp/generator_ekf_cse.py",
    pysrcs = glob(
        ["test/unit/cpp/*.py"],
        allow_empty = False,
    ),
)

cc_formak_model(
    name = "ekf-no-cse",
    namespace = "no_cse",
    pydeps = [],
    pymain = "test/unit/cpp/generator_ekf_no_cse.py",
    pysrcs = glob(
        ["test/unit/cpp/*.py"],
        allow_empty = False,
    ),
)

cc_formak_model(
    name = "calibration-model",
    namespace = "unit",
    pydeps = [],
    pymain = "test/unit/cpp/generator_with_calibration.py",
    pysrcs = glob(
        ["test/unit/cpp/*.py"],
        allow_empty = False,
    ),
)

cc_formak_model(
    name = "calibration-ekf",
    namespace = "unit",
    pydeps = [],
    pymain = "test/unit/cpp/generator_ekf_with_calibration.py",
    pysrcs = glob(
        ["test/unit/cpp/*.py"],
        allow_empty = False,
    ),
)

cc_test(
    name = "unit-cpp-extended-kalman-filter-test",
    srcs = ["test/unit/cpp/ExtendedKalmanFilter_test.cpp"],
    deps = [
        ":simple-ekf",
    ] + CC_TEST_DEPS,
)

cc_test(
    name = "unit-cpp-basic-block",
    srcs = ["test/unit/cpp/BasicBlock_test.cpp"],
    deps = [
        ":ekf-cse",
        ":ekf-no-cse",
    ] + CC_TEST_DEPS,
)

cc_test(
    name = "unit-cpp-model-test",
    srcs = [
        "test/unit/cpp/Model_test.cpp",
    ],
    deps = [
        ":simple-model",
    ] + CC_TEST_DEPS,
)

cc_test(
    name = "unit-cpp-calibration-model-test",
    srcs = [
        "test/unit/cpp/ModelWithCalibration_test.cpp",
    ],
    deps = [
        ":calibration-model",
    ] + CC_TEST_DEPS,
)

cc_test(
    name = "unit-cpp-calibration-ekf-test",
    srcs = [
        "test/unit/cpp/ExtendedKalmanFilterWithCalibration_test.cpp",
    ],
    deps = [
        ":calibration-ekf",
    ] + CC_TEST_DEPS,
)

cc_test(
    name = "integration-cpp-model-test",
    srcs = ["test/formak/Model_test.cpp"],
    deps = [
        ":simple-model",
        ":simple-ekf",
        "//cpp:utils-stats",
        "@rapidcheck",
    ] + CC_TEST_DEPS,
)

cc_test(
    name = "integration-cpp-extended-kalman-filter-test",
    srcs = ["test/formak/ExtendedKalmanFilter_test.cpp"],
    deps = [
        ":simple-model",
        ":simple-ekf",
        "//cpp:utils-stats",
        "@rapidcheck",
    ] + CC_TEST_DEPS,
)
