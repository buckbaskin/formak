from py_to_cpp_codegen import selector

import sys
import pytest


def test_true_case():
    assert selector(True) == 2


def test_false_case():
    assert selector(False) == 3


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv[1:]))
