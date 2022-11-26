import sympy

from sympy import init_printing, symbols

init_printing(use_unicode=True)

a, b, c, d = symbols(["a", "b", "c", "d"])

example_expr = (a + b) * (c + d)


def match_Add(expr):
    return expr.func == sympy.core.add.Add


def match_symbolic_Add(expr):
    if expr.func == sympy.core.add.Add:
        for arg in expr.args:
            if arg.func == sympy.core.symbol.Symbol:
                return "a" in str(arg.name)

    return False


def visit_sympy_expr(expr, matcher, base=None):
    if base is None:
        base = []
    print("visiting expr at %s" % (base,))

    if matcher(expr):
        yield base, expr

    for idx, arg in enumerate(expr.args):
        for result in visit_sympy_expr(arg, matcher, base + [idx]):
            yield result


matches = list(visit_sympy_expr(example_expr, match_symbolic_Add))

print("Matches")
print(matches)
