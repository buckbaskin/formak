from __future__ import annotations

from typing import TypeVar, Generic

class Symbol(object):
    def __init__(self, name):
        self.name = name
    def __add__(self, rhs):
        return Symbol('%s+%s' % (self.name, rhs.name))
    def __mul__(self, rhs):
        return Symbol('%s*%s' % (self.name, rhs.name))
    def __truediv__(self, rhs):
        try:
            return Symbol('%s/%s' % (self.name, rhs.name))
        except AttributeError:
            return Symbol('%s/%s' % (self.name, rhs))

    def __repr__(self):
        return 'Symbol(%s)' % (self.name,)

Meter = TypeVar('Meter')
Second = TypeVar('Second')

class Unit(Symbol, Generic[Meter, Second]):
    def __init__(self, name) -> None:
        super().__init__(name)

    def __add__(self, rhs: Unit[Meter, Second]) -> Unit[Meter, Second]:
        return super().__add__(rhs)

    def __mul__(self, rhs: Unit[Meter, Second]) -> Unit[Meter, Second]:
        return super().__mul__(rhs)

dt = Unit[0,1]('dt') # Unit[0,1]
accel = Unit[1,-2]('accel') # Unit[1,-2]
jerk = Unit[1, -3]('jerk') # Unit[1,-3]

vel = accel * dt + jerk * (dt * dt) / 2
print('vel', vel)

position = Symbol('position') # Unit[1,0]

position = vel + position # Expect to fail here (velocity not the same units as position)
print('position', position)
