from collections import deque
import sympy
from sympy import Symbol, sin, cos, expand_trig
from graphviz import Digraph

def make_expr():
    x = Symbol('x')
    expr_v1= sin(2*x) + cos(2*x)
    expr_v2 = expand_trig(expr_v1)
    return expr_v1, expr_v2

def render_expr(expr):
    if len(expr.args) == 0:
        if expr.func == sympy.core.numbers.One:
            return 'One'
        elif expr.func == sympy.core.numbers.Zero:
            return 'Zero'
        elif expr.func == sympy.core.numbers.NegativeOne:
            return '-One'
        elif expr.func == sympy.core.numbers.Integer:
            return str(expr)
        elif expr.func == sympy.core.symbol.Symbol:
            return str(expr)
        else:
            raise ValueError('what do unknown? %s' % (expr,))

    else:
        if expr.func == sympy.core.add.Add:
            return '+'
        elif expr.func == sympy.core.mul.Mul:
            return '*'
        elif isinstance(expr.func, sympy.core.function.FunctionClass):
            return str(expr.func)
        else:
            raise ValueError(f'what do with args? expr {expr}, type {type(expr.func)} func {expr.func}')


def enumerate_expression(expr, root=None):
    if root is None:
        print('enumerating:', expr)
        root = deque()

    child_ids = []

    root.append(render_expr(expr))

    for child in expr.args:
        for child_id, child_expr, child_children in enumerate_expression(child, root):
            yield child_id, child_expr, child_children
            if child_expr == child:
                # This is false when yielding a grand-child (child's child)
                child_ids.append(child_id)

    id_ = 'root|' + '|'.join(root)
    yield id_, expr, child_ids


    root.pop()

def main():

    v1, v2 = make_expr()

    dot = Digraph(comment='Compute Graph')

    for id_, expr, child_ids in enumerate_expression(v1):
        dot.node(id_, render_expr(expr))
        for child_id in child_ids:
            dot.edge(id_, child_id, constraint='true')

    dot.render(view=True)

if __name__ == "__main__":
    main()
