## Start Selenium-based load
# https://github.com/SeleniumHQ/selenium/blob/c67d0f1dfc49c7f9f6938ac64ae676a348defebd/py/BUILD.bazel
load("@rules_pkg//pkg:mappings.bzl", "pkg_files", "strip_prefix")
load("@rules_pkg//pkg:pkg.bzl", "pkg_tar")
load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load("@rules_python//python:packaging.bzl", "py_package", "py_wheel")
load("@pip_deps//:requirements.bzl", "requirement")
load("//py:defs.bzl", "py_test_suite")
load("@rules_python//python:pip.bzl", "compile_pip_requirements")
# TODOs
# - exporting as a pip package for non-bazel users
## End Selenium-based load

FORMAK_VERSION = "0.0.1-alpha00"

TEST_DEPS = [
    requirement("pytest"),
    requirement("pytest-mock"),
]

py_library(
    name = "formak",
    srcs = glob(
        ["formak/**/*.py"],
        exclude = ["generate.py"],
    ),
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [
        requirement("sympy"),
        requirement("scipy"),
        requirement("numpy"),
        requirement("numba"),
        requirement("hypothesis"),
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

# This is taken verbatim from Selenium, %s/selenium/formak/g
# pkg_files(
#     name = "formak-sdist-pkg",
#     srcs = [
#         "CHANGES.md",
#         "MANIFEST.in",
#         "README.md",
#         "setup.py",
#         ":formak-pkg",
#     ],
#     strip_prefix = strip_prefix.from_pkg(),
# )

# This is taken verbatim from Selenium, %s/selenium/formak/g
# pkg_tar(
#     name = "formak-sdist",
#     srcs = [":formak-sdist-pkg"],
#     extension = "tar.gz",
#     mode = "0644",
#     package_file_name = "formak-%s.tar.gz" % FORMAK_VERSION,
# )

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
    name = "unit",
    size = "small",
    srcs = glob(
        [
            "test/unit/**/*.py",
        ],
        allow_empty = False,
    ),
    deps = [
        ":init-tree",
        ":formak",
    ] + TEST_DEPS,
)

py_test_suite(
    name = "integration",
    size = "small",
    srcs = glob([
        "test/formak/**/*.py",
    ]),
    deps = [
        ":init-tree",
        ":formak",
    ] + TEST_DEPS,
)
