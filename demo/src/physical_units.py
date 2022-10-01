from __future__ import annotations

from typing import TypeVar, Generic
from types import GenericAlias

Meter = TypeVar('Meter')
Second = TypeVar('Second')

def make_unit_subtype(physics_unit: str, quantity: int):
    return TypeVar('%s^%d' % (physics_unit, quantity,))

class Unit():
    def __class_getitem__(cls, key: Tuple[int]):
        meters, seconds = key
        result = GenericAlias(Unit, (make_unit_subtype('Meter', meters), make_unit_subtype('Seconds', seconds)))

        setattr(result, 'meters', meters)
        result.meters = meters
        result.seconds = seconds

        return result

    def __repr__(self):
        return 'Unit[%d,%d](%s)' % (self.meters, self.seconds, self.name,)


    def __init__(self, name) -> None:
        self.name = name

    def __add__(self, rhs: Unit[self.meters, self.seconds]) -> Unit[self.meters, self.seconds]:
        return Unit[self.meters, self.seconds]('%s+%s' % (self.name, rhs.name))
    def __mul__(self, rhs):
        return Unit[self.meters + rhs.meters, self.seconds + rhs.seconds]('%s*%s' % (self.name, rhs.name))
    def __truediv__(self, rhs):
        try:
            return Unit[self.meters - rhs.meters, self.seconds - rhs.seconds]('%s*%s' % (self.name, rhs.name))
        except AttributeError:
            assert not isinstance(rhs, Unit)
            return Unit[self.meters, self.seconds]('%s*%s' % (self.name, rhs))


dt = Unit[0,1]('dt') # Unit[0,1]
accel = Unit[1,-2]('accel') # Unit[1,-2]
jerk = Unit[1, -3]('jerk') # Unit[1,-3]

print('a', accel)
print('dt', dt)
print('a * dt', accel * dt)

vel = accel * dt + jerk * (dt * dt) / 2
print('vel', vel)

position = Unit[1,0]('position') # Unit[1,0]

position = vel + position # Expect to fail here (velocity not the same units as position)
print('position', position)
