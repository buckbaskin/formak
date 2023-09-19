from collections import deque
import sympy
from sympy import Symbol, sin, cos, expand_trig, cse
from graphviz import Digraph


def make_expr():
    x = Symbol("x")
    expr_v1 = sin(2 * x) + cos(2 * x)
    expr_v2 = expand_trig(expr_v1)
    return expr_v1, expr_v2


def render_expr(expr):
    if len(expr.args) == 0:
        if expr.func == sympy.core.numbers.One:
            return "One"
        elif expr.func == sympy.core.numbers.Zero:
            return "Zero"
        elif expr.func == sympy.core.numbers.NegativeOne:
            return "-One"
        elif expr.func == sympy.core.numbers.Integer:
            return str(expr)
        elif expr.func == sympy.core.symbol.Symbol:
            return str(expr)
        else:
            raise ValueError("what do unknown? {}".format(expr))

    else:
        if expr.func == sympy.core.add.Add:
            return "+"
        elif expr.func == sympy.core.mul.Mul:
            return "*"
        elif expr.func == sympy.core.power.Pow:
            return "Pow"
        elif isinstance(expr.func, sympy.core.function.FunctionClass):
            return str(expr.func)
        else:
            raise ValueError(
                f"what do with args? expr {expr}, type {type(expr.func)} func {expr.func}"
            )


def enumerate_expression(expr, subgraph_name="<>", *, path=None, approx_cse=False):
    if path is None:
        print("enumerating:", expr)
        path = deque()

    child_ids = []

    path.append(render_expr(expr))

    for idx, child in enumerate(expr.args):
        path.append(str(idx))

        for child_id, child_expr, child_children in enumerate_expression(
            child, subgraph_name, path=path, approx_cse=approx_cse
        ):
            yield child_id, child_expr, child_children
            if child_expr == child:
                # This is false when yielding a grand-child (child's child)
                child_ids.append(child_id)

        path.pop()

    if len(expr.args) > 0 or not approx_cse:
        id_ = f"{subgraph_name}|" + "|".join(path)
    else:
        id_ = path[-1]

    yield id_, expr, child_ids

    path.pop()


def enumerate_multiple(exprs, subgraph_name="<>", *, path=None):
    for expr in exprs:
        yield from enumerate_expression(expr, subgraph_name, path=path, approx_cse=True)


def main():

    v1, v2 = make_expr()

    dot = Digraph(comment="Compute Graph")

    for id_, expr, child_ids in enumerate_expression(v1, "v1"):
        dot.node(id_, render_expr(expr))
        for child_id in child_ids:
            dot.edge(id_, child_id, constraint="true")

    for id_, expr, child_ids in enumerate_expression(v2, "v2"):
        dot.node(id_, render_expr(expr))
        for child_id in child_ids:
            dot.edge(id_, child_id, constraint="true")

    replacements, reduced_exprs = cse(v1)
    print("v1", v1)
    print("cse", replacements, reduced_exprs)
    for id_, expr in replacements:
        print("replacements", id_, "->", expr, "->", [])
        for id_2, expr_2, child_ids_2 in enumerate_expression(
            expr, "", approx_cse=True
        ):
            print("-- replacements", id_2, "->", expr_2, "->", [])
            dot.node(id_2, render_expr(expr_2), constraint="true")
            for child_id in child_ids_2:
                dot.edge(id_2, child_id, constraint="true")

    # take advantage of fall through and depth-first traversal
    dot.node(str(id_), str(id_))
    print("Extra edge:", str(id_), "->", id_2)
    dot.edge(str(id_), id_2, constraint="true")

    for id_, expr, child_ids in enumerate_multiple(reduced_exprs, "v1_cse"):
        print("reduced", id_, "->", expr, "->", child_ids)
        dot.node(id_, render_expr(expr))
        for child_id in child_ids:
            dot.edge(id_, child_id, constraint="true")

    dot.render(view=True)


if __name__ == "__main__":
    main()
