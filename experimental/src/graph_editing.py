import sympy

from collections import namedtuple
from sympy import Matrix, init_printing, pprint, symbols, shape, srepr
from sympy.matrices.dense import matrix_multiply_elementwise

init_printing(use_unicode=True)

big_num = Matrix([[1, -1], [3, 4], [0, 2]])

pprint(big_num)

big_sym = Matrix([symbols(["a", "b"]), symbols(["c", "d"]), symbols(["e", "f"])])

pprint(big_sym)

print("add")
pprint(big_num + big_sym)

print("mul")
pprint(matrix_multiply_elementwise(big_num, big_sym))

print("matmul? num * sym.T because shapes")
pprint(big_num * big_sym.T)

Assign = namedtuple("Assign", ["target", "value"])
Add = namedtuple("Add", ["left", "right"])
Number = namedtuple("Number", ["value"])
Symbol = namedtuple("Symbol", ["name"])


def expr_to_3addr(expr, counter=0):
    if len(expr.args) == 0:
        if expr.func == sympy.core.numbers.One:
            counter += 1
            yield Assign("tempname%d" % (counter,), Number(1))
        elif expr.func == sympy.core.numbers.Zero:
            counter += 1
            yield Assign("tempname%d" % (counter,), Number(0))
        elif expr.func == sympy.core.numbers.NegativeOne:
            counter += 1
            yield Assign("tempname%d" % (counter,), Number(-1))
        elif expr.func == sympy.core.numbers.Integer:
            counter += 1
            yield Assign("tempname%d" % (counter,), Number(int(expr.evalf())))
        elif expr.func == sympy.core.symbol.Symbol:
            counter += 1
            yield Assign("tempname%d" % (counter,), Symbol(expr.name))
        else:
            raise ValueError("terminal %s %s" % (expr.func, expr.args))
    else:
        last_child_generated = {}
        for idx, arg in enumerate(expr.args):
            for child_generated_3addr in expr_to_3addr(arg, counter):
                last_child_generated[idx] = child_generated_3addr

                counter += 1
                yield child_generated_3addr

        if expr.func == sympy.core.add.Add:
            counter += 1
            final_op = Assign(
                "tempname%d" % (counter,),
                Add(
                    *[last_child_generated[idx].target for idx in range(len(expr.args))]
                ),
            )

            yield final_op
        else:
            raise ValueError("operator %s" % (expr.func,))


def sympy_matrix_to_compute_graph(m):
    assert isinstance(m, Matrix)

    rows, cols = shape(m)

    end_nodes = []

    for row in range(rows):
        for col in range(cols):
            end_nodes.append(Assign("matrix[%d,%d]" % (row, col), m[row, col]))

    the_whole_3addr = []

    for node in end_nodes:
        target, value = node
        the_whole_3addr.extend(expr_to_3addr(value))

        last_value_assign = the_whole_3addr[-1].target

        the_whole_3addr.append(Assign(target, last_value_assign))

    print("=== the_whole_3addr ===")
    for idx, element in enumerate(the_whole_3addr):
        print(idx, element)
    1 / 0


sympy_matrix_to_compute_graph(big_num + big_sym)
