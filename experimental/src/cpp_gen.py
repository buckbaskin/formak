from sympy import symbols, Eq, Matrix

x, y, z = symbols(["x", "y", "z"])
a, b, c, r = symbols(["a", "b", "c", "r"])

result = a * pow(x, 2) + b * x + c * pow(x, 2)

print("Starting".ljust(20), result)

from sympy import simplify

result2 = simplify(result)

print("Simplified".ljust(20), result2)

from sympy.utilities.codegen import CCodeGen, make_routine

gen = CCodeGen(project="formaK", cse=False)
prefix = "prefix"

with open("generated/ccodegen_output_cse_False.cpp", "w") as f:
    gen.dump_c(
        [make_routine("result", result), make_routine("result2", result2)], f, prefix
    )

gen = CCodeGen(project="formaK", cse=True)
prefix = "prefix"

routines = [
    make_routine("result", result),
    make_routine("result2", result2),
    make_routine("fcn", [x * y, Eq(a, 1), Eq(r, x + r), Matrix([[x, 2]])]),
]

with open("generated/ccodegen_output_cse_True.h", "w") as h:
    gen.dump_h(routines, h, prefix)
with open("generated/ccodegen_output_cse_True.cpp", "w") as c:
    gen.dump_c(
        routines,
        c,
        prefix,
    )
