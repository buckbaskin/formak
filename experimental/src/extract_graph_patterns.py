import sympy

from collections import namedtuple
from sympy import Matrix, init_printing, pprint, symbols, shape, srepr
from sympy.matrices.dense import matrix_multiply_elementwise

init_printing(use_unicode=True)

a, b, c, d = symbols(['a', 'b', 'c', 'd'])

example_expr = (a + b) * (c + d)

global_id_counter = 0

def visit_sympy_expr(expr, base=None):
    if base is None:
        base = []
    print('visiting expr at %s' % (base,))

    if len(expr.args) == 0:
        if expr.func == sympy.core.symbol.Symbol:
            print("Symbol doesn't match Add pattern")
        else:
            raise ValueError('terminal %s %s' % (expr.func, expr.args,))
    else:
        if expr.func == sympy.core.add.Add:
            # match Add
            yield base, expr

            for idx, arg in enumerate(expr.args):
                for result in visit_sympy_expr(arg, base + [idx]):
                    yield result
        elif expr.func == sympy.core.mul.Mul:
            for idx, arg in enumerate(expr.args):
                for result in visit_sympy_expr(arg, base + [idx]):
                    yield result
        else:
            raise ValueError('operator %s %s' % (expr.func, expr.args,))

matches = list(visit_sympy_expr(example_expr))

print('Matches')
print(matches)
