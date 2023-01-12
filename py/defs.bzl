load("//py/private:pytest.bzl", _pytest_test = "pytest_test")
load("//py/private:suite.bzl", _py_test_suite = "py_test_suite")
load("//py/private:suite.bzl", _cc_test_suite = "cc_test_suite")

pytest_test = _pytest_test
py_test_suite = _py_test_suite
cc_test_suite = _cc_test_suite
