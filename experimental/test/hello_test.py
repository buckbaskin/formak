from hello_greet import magic

import sys
import pytest


def test_answer():
    assert magic() == 1


if __name__ == "__main__":
    sys.exit(pytest.main(sys.argv[1:]))
