load("//py/private:pytest.bzl", _pytest_test = "pytest_test")
load("//py/private:suite.bzl", _cc_test_suite = "cc_test_suite", _py_test_suite = "py_test_suite")
load("//py/private:formak_gen.bzl", _cc_formak_model = "cc_formak_model")

pytest_test = _pytest_test
py_test_suite = _py_test_suite
cc_test_suite = _cc_test_suite
cc_formak_model = _cc_formak_model
