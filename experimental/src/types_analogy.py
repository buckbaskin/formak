# Meters, Seconds, Kilograms
Unit = Tuple[int, int, int]

Meters = (1,0,0)
Seconds = (0,1,0)
Kilograms = (0,0,1)
MetersPerSecond = (1, -1, 0)
MetersPerSecondSquared = (1, -2, 0)

def mul(left, right):
    def mul_impl():
        for ul, ur in zip(left, right):
            yield ul, ur

    return tuple(mul_impl(left, right))

# F = ma
Newtons = mul(Kilograms, MetersPerSecondSquared)
