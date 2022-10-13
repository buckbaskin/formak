from __future__ import annotations
from typing import Tuple, Union

# Meters, Seconds, Kilograms
BaseUnit = Tuple[int, int, int]


class UnaryOp(object):
    def __init__(self, left: Unit) -> None:
        self.left = left


class Mul(object):
    def __init__(self, left: Unit, right: Unit) -> None:
        self.left = left
        self.right = right


class Div(object):
    def __init__(self, left: Unit, right: Unit) -> None:
        self.left = left
        self.right = right


Unit = Union[BaseUnit, UnaryOp, Mul]

Radians = (0, 0, 0)
Meters = (1, 0, 0)
Seconds = (0, 1, 0)
Kilograms = (0, 0, 1)
MetersPerSecond = (1, -1, 0)


def collapse(unit: Unit) -> BaseUnit:
    # if isinstance(unit, BaseUnit):
    if isinstance(unit, tuple) and len(unit) == 3:
        return unit
    if isinstance(unit, UnaryOp):
        return collapse(unit.left)
    if isinstance(unit, Mul):
        return tuple(
            (ul + ur for ul, ur in zip(collapse(unit.left), collapse(unit.right)))
        )
    if isinstance(unit, Div):
        return tuple(
            (ul - ur for ul, ur in zip(collapse(unit.left), collapse(unit.right)))
        )


def neg(left: Unit) -> UnaryOp:
    return UnaryOp(left)


def mul(left: Unit, right: Unit) -> Mul:
    return Mul(left, right)


def div(left: Unit, right: Unit) -> Div:
    return Div(left, right)


def sin(angle: Radians) -> Radians:
    assert collapse(angle) == Radians
    return angle


def add(left: Unit, right: Unit) -> BaseUnit:
    assert collapse(left) == collapse(right)
    return collapse(left)


def eq(left: Unit, right: Unit) -> None:
    assert collapse(left) == collapse(right)


### Test Cases ###

MetersPerSecondSquared = div(MetersPerSecond, Seconds)

# F = ma
Newtons = collapse(mul(Kilograms, MetersPerSecondSquared))
assert Newtons == (1, -2, 1)

try:
    sin(Meters)
    raise ValueError("Expected AssertionError")
except AssertionError:
    pass

try:
    add(Meters, Kilograms)
    raise ValueError("Expected AssertionError")
except AssertionError:
    pass

eq((1, 0, 1), mul(Meters, neg(Kilograms)))

print("Done, Passing")
