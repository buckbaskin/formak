import sys
import numpy as np
import pstats
from pstats import SortKey
import cProfile
from collections import defaultdict
from datetime import datetime
from sympy import (
    Symbol,
    Rational,
    Quaternion,
    simplify,
    Matrix,
    symbols,
    Expr,
    parse_expr
)


def decomposing_print(expr) -> None:
    print({"func": expr.func, "args": expr.args})


def bottoms_up_traversal(expr, level=0) -> None:
    # print('Descending', level, expr.func)

    for arg in expr.args:
        yield from bottoms_up_traversal(arg, level - 1)

    yield expr
    # print('Ascending', level, expr.func)

def structured_simplify(expr, *, level=0):
    '''
    Simplify bottoms up in an attempt to speed up the result
    '''
    if isinstance(expr, Rational):
        return expr
    if isinstance(expr, Symbol):
        return expr

    simplified_args = []
    for arg in expr.args:
        simplified_args.append(structured_simplify(arg, level=level - 1))

    print(expr)
    print(type(expr))
    print(expr.args)
    print(simplified_args)
    expr = expr.func(*simplified_args)

    return simplify(expr)



def make_a_slow_expr(*, debug=False):
    print('make_a_slow_expr')
    reference = Matrix(
        [symbols(["a", "b", "c"]), symbols(["d", "e", "f"]), symbols(["g", "h", "i"])]
    )

    r = Quaternion.from_rotation_matrix(reference)

    r_mat = r.to_rotation_matrix()

    # for i in range(3):
    #     for j in range(3):
    #         print(i, j, "\n", r_mat[i, j])

    time_history = {}
    time_history['simplify'] = defaultdict(list)
    time_history['structured_simplify'] = defaultdict(list)

    for idx, expr in enumerate(bottoms_up_traversal(r_mat[0, 0])):
        start = datetime.now()

        simplify(expr)

        end = datetime.now()

        simplify_time = (end - start).total_seconds()
        assert simplify_time >= 0.0
        time_history['simplify'][expr].append(simplify_time)

        structured_simplify_time = -1.0

        if debug:
            print("Index", idx, "\n", simplify_time, structured_simplify_time)

        if simplify_time > 1.0 or structured_simplify_time > 1.0:
            if debug:
                print("Big Expression\n", expr)

        if simplify_time > 10.0 or structured_simplify_time > 10.0:
            if debug:
                print("Dump Timing Summary")
                for key, times in sorted(list(time_history['simplify'].items()), key=lambda t: (np.max(t[1]), str(t[0]))):
                    print(key, ">>>>>")
                    print("    |", len(times), np.average(times), np.max(times))
                    times = time_history['structured_simplify'][key]
                    print("    |", len(times), np.average(times), np.max(times))
            return expr

def main():
    print('Python Version %s' % (sys.version_info,))
    1/0
    # expr = make_a_slow_expr()
    expr = parse_expr("(a/4 + e/4 + i/4 - (-a - e + i + (a*e*i - a*f*h - b*d*i + b*f*g + c*d*h - c*e*g)**(1/3))*sign(-b + d)**2/4 - (-a + e - i + (a*e*i - a*f*h - b*d*i + b*f*g + c*d*h - c*e*g)**(1/3))*sign(c - g)**2/4 + (a - e - i + (a*e*i - a*f*h - b*d*i + b*f*g + c*d*h - c*e*g)**(1/3))*sign(-f + h)**2/4 + (a*e*i - a*f*h - b*d*i + b*f*g + c*d*h - c*e*g)**(1/3)/4)/(a/4 + e/4 + i/4 + (-a - e + i + (a*e*i - a*f*h - b*d*i + b*f*g + c*d*h - c*e*g)**(1/3))*sign(-b + d)**2/4 + (-a + e - i + (a*e*i - a*f*h - b*d*i + b*f*g + c*d*h - c*e*g)**(1/3))*sign(c - g)**2/4 + (a - e - i + (a*e*i - a*f*h - b*d*i + b*f*g + c*d*h - c*e*g)**(1/3))*sign(-f + h)**2/4 + (a*e*i - a*f*h - b*d*i + b*f*g + c*d*h - c*e*g)**(1/3)/4)")
    print('Slow Expression')
    print(expr)

    start = datetime.now()
    print('Begin Profiling')
    cProfile.runctx('simplify(expr)', globals=globals(), locals={'expr': expr}, filename='simplify_pstats')
    print('End Profiling')
    end = datetime.now()
    print('Profiling took about', (end - start).total_seconds(), 'seconds')

    profile = pstats.Stats('simplify_pstats')
    profile.strip_dirs().sort_stats(SortKey.CUMULATIVE).print_stats(10)


if __name__ == "__main__":
    main()
