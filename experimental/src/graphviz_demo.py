from collections import deque
import sympy
from sympy import Symbol, sin, cos, expand_trig
from graphviz import Digraph

def make_expr():
    x = Symbol('x')
    expr_v1= sin(2*x) + cos(2*x)
    expr_v2 = expand_trig(expr_v1)
    return expr_v1, expr_v2

def enumerate_expression(expr, root=None):
    if root is None:
        print('enumerating:', expr)
        root = deque()

    if len(expr.args) == 0:
        if expr.func == sympy.core.numbers.One:
            root.append('One')
        elif expr.func == sympy.core.numbers.Zero:
            root.append('Zero')
        elif expr.func == sympy.core.numbers.NegativeOne:
            root.append('-One')
        elif expr.func == sympy.core.numbers.Integer:
            root.append(str(expr))
        elif expr.func == sympy.core.symbol.Symbol:
            root.append(str(expr))
        else:
            raise ValueError('what do unknown? %s' % (expr,))

    else:
        print(expr.func, expr.args)
        if expr.func == sympy.core.add.Add:
            root.append('+')
        elif expr.func == sympy.core.mul.Mul:
            root.append('*')
        elif isinstance(expr.func, sympy.core.function.FunctionClass):
            root.append(str(expr.func))
        else:
            raise ValueError(f'what do with args? expr {expr}, type {type(expr.func)} func {expr.func}')

    id_ = 'root|' + '|'.join(root)
    yield id_, expr

    for child in expr.args:
        yield from enumerate_expression(child, root)

    root.pop()

def main():

    v1, v2 = make_expr()

    for id_, expr in enumerate_expression(v1):
        print(id_)

    dot = graphviz.Digraph(comment='The Round Table')
    dot.node('A', 'King Arthur')
    dot.node('B', 'Sir Bedevere the Wise')
    dot.node('L', 'Sir Lancelot the Brave')
    dot.edge('B', 'L', constraint='true')

    dot.render(view=True)

if __name__ == "__main__":
    main()
