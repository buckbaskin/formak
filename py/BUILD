## Start Selenium-based load
# https://github.com/SeleniumHQ/selenium/blob/c67d0f1dfc49c7f9f6938ac64ae676a348defebd/py/BUILD.bazel
load("@rules_pkg//pkg:mappings.bzl", "pkg_files", "strip_prefix")
load("@rules_pkg//pkg:pkg.bzl", "pkg_tar")
load("@rules_python//python:defs.bzl", "py_binary", "py_library")
load("@rules_python//python:packaging.bzl", "py_package", "py_wheel")
load("@py_dev_requirements//:requirements.bzl", "requirement")
load("//common:defs.bzl", "copy_file")
load("//py:defs.bzl", "py_test_suite")
load("//py/private:browsers.bzl", "BROWSERS")
load("@rules_python//python:pip.bzl", "compile_pip_requirements")
# TODOs
# - investigate pip deps process:
#   - defining requirements.txt
#   - version pinning
#   - defining a dependency
# - py_test_suite?
# - exporting as a pip package for non-bazel users
# - generate py_binary rule
## End Selenium-based load

# TODO(buck): define this more globally for reference in other packages?
FORMAK_VERSION = "0.0.1-alpha00"

# Superceded by Selenium rule
# load("@rules_python//python:defs.bzl", "py_binary")
# Name Conflict with Selenium rule
# load("@pip_deps//:requirements.bzl", "requirement")

# Selenium pattern for specifying common test deps
TEST_DEPS = [
    requirement("pytest"),
    requirement("pytest-mock"),  # kinda expect this one to fail in some way because it's not in requirements
]

## Selenium Example: Copy files from other packages to be available here
# https://github.com/SeleniumHQ/selenium/blob/c67d0f1dfc49c7f9f6938ac64ae676a348defebd/py/BUILD.bazel
# copy_file(
#     name = "get-attribute",
#     src = "//javascript/webdriver/atoms:get-attribute.js",
#     out = "selenium/webdriver/remote/getAttribute.js",
# )

py_library(
    name = "formak",
    srcs = glob(
        "formak/**/*.py",
        exclude = ["generate.py"],
    ),
    imports = ["."],
    visibility = ["//visibility:public"],
    deps = [requirement("sympy")],
)

# TODO(buck): Write up the difference between py_library and py_package

py_package(
    name = "formak-pkg",
    packages = ["py.formak", "py.formak.ui", "py.formak.python", "py.formak.cpp"]
    deps = [":formak"],
        )

# This is taken verbatim from Selenium, %s/selenium/formak/g
pkg_files(
name = "formak-sdist-pkg",
srcs = [
        "CHANGES",
        "MANIFEST.in",
        "README.rst",
        "setup.py",
        ":formak-pkg",
    ],
    strip_prefix = strip_prefix.from_pkg(),
)
        )

# This is taken verbatim from Selenium, %s/selenium/formak/g
pkg_tar(
    name = "formak-sdist",
    srcs = [":formak-sdist-pkg"],
    extension = "tar.gz",
    mode = "0644",
    package_file_name = "formak-%s.tar.gz" % SE_VERSION,
)

# This is taken verbatim from Selenium
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

# I think this might be the sauce for making pytest work more nicely
py_library(
    name = "init-tree",
    testonly = True,
    srcs = [
        "conftest.py",
        "test/__init__.py",
        "test/formak/__init__.py",
        # "test/formak/webdriver/__init__.py",
        # "test/formak/webdriver/chrome/__init__.py",
        # "test/formak/webdriver/common/__init__.py",
        # "test/formak/webdriver/common/conftest.py",
        # "test/formak/webdriver/common/network.py",
        # "test/formak/webdriver/common/webserver.py",
        # "test/formak/webdriver/firefox/__init__.py",
        # "test/formak/webdriver/firefox/conftest.py",
        # "test/formak/webdriver/marionette/__init__.py",
        # "test/formak/webdriver/marionette/conftest.py",
        # "test/formak/webdriver/safari/conftest.py",
        # "test/formak/webdriver/support/__init__.py",
        # "test/formak/webdriver/support/conftest.py",
    ],
    data = [
        "pytest.ini",
        "setup.cfg",
        # "test/formak/webdriver/common/test_file.txt",
        # "test/formak/webdriver/common/test_file2.txt",
    ],
    imports = ["."],
    deps = [
        # ":webserver", # server used to respond to browser requests in selenium testing
    ],
)

py_test_suite(
    name = "unit",
    size = "small",
    srcs = glob([
        "test/unit/**/*.py",
    ]),
    # args = [
    #     "--instafail",
    # ],
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
    # args = [
    #     "--instafail",
    # ],
    deps = [
        ":init-tree",
        ":formak",
    ] + TEST_DEPS,
)

# Not sure about what this rule is used for
# py_binary(
#     name = "generate",
#     srcs = ["generate.py"],
#     srcs_version = "PY3",
#     deps = [requirement("inflection")],
# )

## Trying alternate Selenium-inspired packaging
# py_library(
#     name = "base",
#     srcs = [
#         "src/formak/__init__.py",
#     ],
#     imports = ["src"],
#     visibility = ["//visibility:private"],
#     deps = [],
# )
#
# py_library(
#     name = "user-interface",
#     srcs = [
#         "src/formak/ui.py",
#     ],
#     imports = ["src"],
#     visibility = ["//visibility:public"],
#     deps = [
#         requirement("sympy"),
#     ],
# )
#
# py_library(
#     name = "python",
#     srcs = [
#         "src/formak/python.py",
#     ],
#     imports = ["src"],
#     visibility = ["//visibility:public"],
#     deps = [],
# )
#
# py_test(
#     name = "user-interface-test-py",
#     srcs = ["test/ui_test.py"],
#     main = "test/ui_test.py",
#     deps = [
#         ":user-interface",
#         requirement("pytest"),
#     ],
# )
# 
# py_test(
#     name = "py-test-py",
#     srcs = ["test/python_test.py"],
#     main = "test/python_test.py",
#     deps = [
#         ":python",
#         ":user-interface",
#         requirement("pytest"),
#     ],
# )
