package(default_visibility = ["//visibility:public"])

## Start Selenium-based load
# https://github.com/SeleniumHQ/selenium/blob/c67d0f1dfc49c7f9f6938ac64ae676a348defebd/py/BUILD.bazel
load("@rules_python//python:defs.bzl", "py_library")
load("@rules_python//python:packaging.bzl", "py_package", "py_wheel")
load("@pip_deps//:requirements.bzl", "requirement")
load("//py:defs.bzl", "PY_TEST_DEPS", "py_test_suite")
load("@rules_python//python:pip.bzl", "compile_pip_requirements")
# TODOs
# - exporting as a pip package for non-bazel users
## End Selenium-based load

FORMAK_VERSION = "0.0.1-alpha00"

# Note: allowed library for external use
py_library(
    name = "problemdefinition",
    srcs = glob(
        ["formak/problemdefinition/*.py"],
        allow_empty = False,
    ),
    imports = ["."],
    deps = [
        "//problemdefinition:design-manager",
        "//problemdefinition:model",
        "//problemdefinition:ui-model-base",
    ],
)

# Note: allowed library for external use
py_library(
    name = "runtime_py",
    srcs = glob(
        ["formak/runtime_py/*.py"],
        allow_empty = False,
    ),
    imports = ["."],
    deps = [
        "//runtime_py:managed-filter",
        "//runtime_py:plotting",
        "//runtime_py:stamped-reading",
    ],
)

# Note: allowed library for external use
py_library(
    name = "frontend",
    srcs = glob(
        ["formak/frontend/*.py"],
        allow_empty = False,
    ),
    imports = ["."],
    deps = [
        "//frontend:model-validation",
    ],
)

py_library(
    name = "backend_cpp",
    srcs = glob(
        ["formak/backend_cpp/*.py"],
        allow_empty = False,
    ),
    imports = ["."],
    deps = [
        "//backend_cpp:compile-ekf",
        "//backend_cpp:compile-model",
        "//backend_cpp:config",
    ],
)

py_library(
    name = "backend_py",
    srcs = glob(
        ["formak/backend_py/*.py"],
        allow_empty = False,
    ),
    imports = ["."],
    deps = [
        "//backend_py:compile-ekf",
        "//backend_py:compile-model",
        "//backend_py:config",
        "//backend_py:model",
        "//backend_py:sensor-model",
        "//backend_py:sklearn",
    ],
)

py_library(
    name = "python",
    srcs = ["formak/python.py"],
    imports = ["."],
    deps = [
        ":exceptions",
        "//backend_py:basic-block",
        "//backend_py:compile-ekf",
        "//backend_py:config",
        "//backend_py:extended-kalman-filter",
        "//backend_py:named-covariance",
        "//backend_py:named-vector",
        "//backend_py:sensor-model",
        "//problemdefinition:ui-model-base",
        requirement("sympy"),
        requirement("scipy"),
        requirement("numpy"),
        requirement("scikit-learn"),
    ],
)

# Note: allowed library for external use (for now)
py_library(
    name = "exceptions",
    srcs = ["formak/exceptions.py"],
    imports = ["."],
    deps = [
    ],
)

# Note: allowed library for external use
py_library(
    name = "reference-models",
    srcs = glob(
        ["formak/reference_models/*.py"],
        allow_empty = False,
    ),
    imports = ["."],
    deps = [
        ":problemdefinition",
        ":python",
        requirement("sympy"),
    ],
)

# TODO: remove deps here that aren't intended for general use, measured by deps not used in feature tests or demos
py_library(
    name = "formak",
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [
        ":backend_cpp",
        ":backend_py",
        ":frontend",
        ":problemdefinition",
        ":python",
        ":reference-models",
        ":runtime_py",
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
        ":python",
        "//problemdefinition:model",
        "//backend_py:compile-model",
        "//backend_py:sklearn",
    ] + PY_TEST_DEPS,
)

py_test_suite(
    name = "reference-models-test",
    timeout = "moderate",
    srcs = glob(
        [
            "test/unit/reference_models/*.py",
        ],
        allow_empty = False,
    ),
    deps = [
        ":init-tree",
        ":reference-models",
        "//runtime_py:plotting",
        "//backend_py:compile-model",
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
