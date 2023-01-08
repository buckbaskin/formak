import argparse

from os import mkdir, walk, scandir
from os.path import dirname, basename
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


def generate_function_bodies(header_location):
    # TODO(buck): replace SympyModel_model_body with sympy model
    # SympyModel_model_body: x*y + x + y + 1
    model = x * y + x + y + 1

    ccode_model = ccode(model)

    # includes header is a single file name (e.g. abc.h) not in some directory structure (e.g. foo/bar/abc.h)
    header_include = basename(header_location)

    return {
        "header_include": header_include,
        "update_body": "_state += 1;",
        "getValue_body": "return _state;",
        "SympyModel_model_body": "return {};".format(ccode_model),
    }


import sys

print(sys.argv)

# python3 $(location src/cpp_gen.py) --headertemplate $(location templates/basic_class.h) --sourcetemplate $(location templates/basic_class.cpp) --header $(location generated/jinja_basic_class.h) --source $(location generated/jinja_basic_class.cpp)
parser = argparse.ArgumentParser(prog="cppgen")
parser.add_argument("--headertemplate")
parser.add_argument("--sourcetemplate")
parser.add_argument("--header")
parser.add_argument("--source")

args = parser.parse_args()
print("args")
print((args.headertemplate, args.header), "\n", (args.sourcetemplate, args.source))

templates_base_path = dirname(args.headertemplate)
assert templates_base_path == dirname(args.sourcetemplate)
print("templates_base_path", templates_base_path)

env = Environment(
    loader=FileSystemLoader(templates_base_path), autoescape=select_autoescape()
)

try:
    header_template = env.get_template("basic_class.h")
    source_template = env.get_template("basic_class.cpp")
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

inserts = generate_function_bodies(args.header)

header_str = header_template.render(**inserts)
source_str = source_template.render(**inserts)

with open(args.header, "w") as header_file:
    with open(args.source, "w") as source_file:
        # Stack indents so an error in either file will write/close both the same way

        print("Writing header arg {}".format(args.header))
        header_file.write(header_str)

        print("Writing source arg {}".format(args.source))
        source_file.write(source_str)
