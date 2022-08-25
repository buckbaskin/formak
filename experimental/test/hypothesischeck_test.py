import pytest
import sys

from hypothesis import given
from hypothesis.strategies import lists, integers


def test_assumption():
    assert [] == []


@given(lists(integers()))
def test_double_reversal(l0):
    l1 = list(reversed(l0))
    l1 = list(reversed(l1))
    assert l0 == l1


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv[1:]))
