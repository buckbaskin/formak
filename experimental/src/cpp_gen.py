import argparse

from os import mkdir, walk, scandir
from os.path import dirname
from sympy import symbols, Eq, Matrix, ccode
from jinja2 import Environment, FileSystemLoader, PackageLoader, select_autoescape
from jinja2.exceptions import TemplateNotFound

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

try:
    mkdir("generated/")
except FileExistsError as fee:
    print("Silencing Error %s" % fee)

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

print("Template Based Custom Insertion")


def generate_function_bodies():
    # TODO(buck): replace SympyModel_model_body with sympy model
    # SympyModel_model_body: x*y + x + y + 1
    model = x * y + x + y + 1

    ccode_model = ccode(model)

    return {
        "update_body": "_state += 1;",
        "getValue_body": "return _state;",
        "SympyModel_model_body": "return {};".format(ccode_model),
    }


import sys

print(sys.argv)

parser = argparse.ArgumentParser(prog="cppgen")
parser.add_argument("--template")
parser.add_argument("--header")
parser.add_argument("--source")

args = parser.parse_args()
print("args")
print(args.template, args.header, args.source)

templates_base_path = dirname(args.template)
print("templates_base_path", templates_base_path)

env = Environment(
    loader=FileSystemLoader(templates_base_path), autoescape=select_autoescape()
)

try:
    template = env.get_template("basic_class.cpp")
except TemplateNotFound:
    print("Debugging TemplateNotFound")
    print("Trying to scandir")
    with scandir(templates_base_path) as it:
        if len(list(it)) == 0:
            print("No Paths in scandir")
            raise

    print("Walking")
    for root, dirs, files in walk(templates_base_path):
        depth = len(root.split("/"))
        print("{}Root: {!s}".format(" " * depth, root))
        for filename in files:
            print("{}  - {!s}".format(" " * depth, filename))
    print("End Walk")
    raise

inserts = generate_function_bodies()

generated_str = template.render(**inserts)

with open(args.source, "w") as f:
    print("Writing source arg {}".format(args.source))
    f.write(generated_str)
