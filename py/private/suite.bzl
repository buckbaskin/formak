load("@rules_python//python:defs.bzl", "py_library")
load("//py/private:pytest.bzl", "pytest_test")

def _is_py_test(file):
    return file.startswith("test_") or file.endswith("_tests.py") or file.endswith("_test.py")

def py_test_suite(name, srcs, size = None, deps = None, python_version = None, imports = None, visibility = None, **kwargs):
    library_name = "%s-py-test-lib" % name

    py_library(
        name = library_name,
        testonly = True,
        srcs = srcs,
        deps = deps,
        imports = imports,
    )

    tests = []
    for src in srcs:
        if _is_py_test(src):
            test_name = "%s-%s" % (name, src)

            tests.append(test_name)

            pytest_test(
                name = test_name,
                size = size,
                srcs = [src],
                deps = [library_name],
                python_version = python_version,
                **kwargs
            )
    native.test_suite(
        name = name,
        tests = tests,
        visibility = visibility,
    )

def _is_cpp_test(file):
    return file.startswith("test_") or file.endswith("_test.cpp")

def cc_test_suite(name, srcs, size = None, deps = None, python_version = None, imports = None, visibility = None, **kwargs):
    tests = []
    for src in srcs:
        if _is_cpp_test(src):
            test_name = "%s-%s" % (name, src)

            tests.append(test_name)

            native.cc_test(
                name = test_name,
                srcs = [src],
                deps = deps,
                **kwargs
            )

    native.test_suite(
        name = name,
        tests = tests,
        visibility = visibility,
    )
