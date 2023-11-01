import numpy as np
from collections import defaultdict
from datetime import datetime
from sympy import (
    Symbol,
    Quaternion,
    simplify,
    Matrix,
    symbols,
    Expr,
)


def decomposing_print(expr) -> None:
    print({"func": expr.func, "args": expr.args})


def bottoms_up_traversal(expr, level=0) -> None:
    # print('Descending', level, expr.func)

    for arg in expr.args:
        yield from bottoms_up_traversal(arg, level - 1)

    yield expr
    # print('Ascending', level, expr.func)


def main():
    reference = Matrix(
        [symbols(["a", "b", "c"]), symbols(["d", "e", "f"]), symbols(["g", "h", "i"])]
    )

    r = Quaternion.from_rotation_matrix(reference)

    r_mat = r.to_rotation_matrix()

    # for i in range(3):
    #     for j in range(3):
    #         print(i, j, "\n", r_mat[i, j])

    time_history = defaultdict(list)

    for idx, expr in enumerate(bottoms_up_traversal(r_mat[0, 0])):
        start = datetime.now()

        simplify(expr)

        end = datetime.now()

        simplify_time = (end - start).total_seconds()
        assert simplify_time >= 0.0
        time_history[expr].append(simplify_time)

        print("Index", idx, "\n", simplify_time)

        if simplify_time > 1.0:
            print("Big Expression\n", expr)

        if simplify_time > 10.0:
            print("Dump Timing Summary")
            for key, times in sorted(list(time_history.items()), key=lambda t: (np.max(t[1]), str(t[0]))):
                print(key, "|")
                print("    |", len(times), np.average(times), np.max(times))
            raise ValueError("Long Simplify")


if __name__ == "__main__":
    main()
