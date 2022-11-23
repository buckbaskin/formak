import sympy

from collections import namedtuple
from sympy import Matrix, init_printing, pprint, symbols, shape, srepr
from sympy.matrices.dense import matrix_multiply_elementwise

init_printing(use_unicode=True)

a, b, c, d = symbols(['a', 'b', 'c', 'd'])

example_expr = (a + b) * (c + d)

global_id_counter = 0

def match_Add(expr):
    return expr.func == sympy.core.add.Add

def visit_sympy_expr(expr, matcher, base=None):
    if base is None:
        base = []
    print('visiting expr at %s' % (base,))

    if matcher(expr):
        yield base, expr

    for idx, arg in enumerate(expr.args):
        for result in visit_sympy_expr(arg, matcher, base + [idx]):
            yield result

matches = list(visit_sympy_expr(example_expr, match_Add))

print('Matches')
print(matches)
