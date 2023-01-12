import argparse
from os import basename, dirname, scandir
from os.path import walk

from jinja2 import Environment, FileSystemLoader, select_autoescape
from jinja2.exceptions import TemplateNotFound
from sympy import ccode


# TODO(buck): data class?
class CppCompileResult:
    def __init__(self, success, header_path=None, source_path=None):
        self.success = success
        self.header_path = header_path
        self.source_path = source_path


def _generate_function_bodies(header_location):
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


def compile(symbolic_model, *, config=None):
    if config is None:
        config = Config()
    elif isinstance(config, dict):
        config = Config(**config)

    parser = argparse.ArgumentParser(prog="generator.py")
    parser.add_argument("--templates", nargs=2)
    parser.add_argument("--header")
    parser.add_argument("--source")

    args = parser.parse_args()

    header_template, source_template = args.templates

    templates_base_path = dirname(header_template)
    assert templates_base_path == dirname(source_template)

    env = Environment(
        loader=FileSystemLoader(templates_base_path), autoescape=select_autoescape()
    )

    try:
        header_template = env.get_template("formak_model.h")
        source_template = env.get_template("formak_model.cpp")
    except TemplateNotFound:
        print("Debugging TemplateNotFound")
        print("Trying to scandir")
        with scandir(templates_base_path) as it:
            if len(list(it)) == 0:
                print("No Paths in scandir")
                raise

        print("Walking")
        for root, _, files in walk(templates_base_path):
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

    return CppCompileResult(
        success=True, header_path=args.header, source_path=args.source
    )
