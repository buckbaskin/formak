# TODO(buck): Get this working for simple Symbol integration, ABC
from __future__ import annotations
from typing import Tuple, Union, Type, Any


class Symbol(object):
    pass


# Meters, Seconds, Kilograms
BaseUnit = Tuple[int, int, int]


def base_unit_cast(u: Union[BaseUnit, Tuple[int, ...]]) -> BaseUnit:
    return (u[0], u[1], u[2])


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


BinOp = Union[Mul, Div]


class TypedSymbolWithUnit(Symbol):
    @staticmethod
    def class_unit() -> BaseUnit:
        raise NotImplementedError()


Unit = Union[TypedSymbolWithUnit, UnaryOp, BinOp]


def type_maker(unit: BaseUnit):
    class TypedSymbolWithUnitImpl(TypedSymbolWithUnit):
        @classmethod
        def __subclasshook__(cls, C):
            print('TypedSymbolWithUnit.__subclasshook__')
            return C.class_unit() == cls.class_unit()

        @staticmethod
        def class_unit() -> BaseUnit:
            return unit

    return TypedSymbolWithUnitImpl


# TODO(buck): With Python 3.10 support, use TypeAlias instead of Any here
Radians: Any = type_maker((0, 0, 0))
Meters: Any = type_maker((1, 0, 0))
Seconds: Any = type_maker((0, 1, 0))
Kilograms: Any = type_maker((0, 0, 1))
MetersPerSecond: Any = type_maker((1, -1, 0))
MetersPerSecondSquared: Any = type_maker((1, -2, 0))


def collapse(unit: Unit) -> BaseUnit:
    # if isinstance(unit, BaseUnit):
    if isinstance(unit, UnaryOp):
        return collapse(unit.left)
    if isinstance(unit, Mul):
        return base_unit_cast(
            tuple(
                (ul + ur for ul, ur in zip(collapse(unit.left), collapse(unit.right)))
            )
        )
    if isinstance(unit, Div):
        return base_unit_cast(
            tuple(
                (ul - ur for ul, ur in zip(collapse(unit.left), collapse(unit.right)))
            )
        )
    if hasattr(unit, "class_unit"):
        return unit.class_unit()
    raise ValueError("collapse(%s, %s)" % (type(unit), unit))

assert isinstance(Meters(), type_maker((1, 0, 0)))
assert issubclass(type_maker((1,0,0)), Meters)
