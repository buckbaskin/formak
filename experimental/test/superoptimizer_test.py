from superoptimizer import superoptimizer, astar


def test_astar():
    result = astar()
    assert result == ["a", "b"]


def test_superoptimizer():
    result = superoptimizer()
    assert result == ["add", "mul"]
