import argparse
from os import scandir, walk
from os.path import dirname
from typing import Any, List, Tuple

from colorama import Fore as cF
from colorama import Style as cS
from colorama import init
from jinja2 import Environment, FileSystemLoader, select_autoescape
from jinja2.exceptions import TemplateNotFound
from sympy import Symbol, ccode, cse

DEFAULT_MODULES = ("scipy", "numpy", "math")


class Config:
    def __init__(
        self,
        common_subexpression_elimination=True,
        python_modules=DEFAULT_MODULES,
    ):
        self.common_subexpression_elimination = common_subexpression_elimination
        self.python_modules = python_modules


# TODO(buck): data class?
class CppCompileResult:
    def __init__(self, success, header_path=None, source_path=None):
        self.success = success
        self.header_path = header_path
        self.source_path = source_path


class BasicBlock:
    def __init__(self, statements: List[Tuple[str, Any]]):
        # should be Tuple[str, sympy expression]
        self._statements = statements

    def compile(self):
        return "\n".join(self._compile_impl())

    def _compile_impl(self):
        # TODO(buck): this ignores a CSE flag in favor of never doing it. Add in the flag
        # TODO(buck): Common Subexpression Elimination supports multiple inputs, so use common subexpression elimination across state calculations
        # Note: The list of statements is ordered and can get CSE or reordered within the block because we know it is straight calculation without control flow (a basic block)
        for varname, expr in self._statements:
            cc_expr = ccode(expr)
            yield "double {varname} = {cc_expr};".format(
                varname=varname,
                cc_expr=cc_expr,
            )


class Model:
    """C++ implementation of the model."""

    def __init__(self, symbolic_model, config):
        # TODO(buck): Enable mypy for type checking
        # TODO(buck): Move all type assertions to either __init__ (constructor) or mypy?
        # assert isinstance(symbolic_model, ui.Model)
        if isinstance(config, dict):
            config = Config(**config)
        assert isinstance(config, Config)

        self.state_size = len(symbolic_model.state)
        self.control_size = len(symbolic_model.control)

        self.arglist_state = sorted(list(symbolic_model.state), key=lambda x: x.name)
        self.arglist_control = sorted(
            list(symbolic_model.control), key=lambda x: x.name
        )
        self.arglist = [symbolic_model.dt] + self.arglist_state + self.arglist_control

        self._impl = BasicBlock(self._translate_impl(symbolic_model))

        self._return = self._translate_return()

    def _translate_impl(self, symbolic_model):
        subs_set = [
            (
                member,
                Symbol("input_state.{}".format(member)),
            )
            for member in self.arglist_state
        ] + [
            (
                member,
                Symbol("input_control.{}".format(member)),
            )
            for member in self.arglist_control
        ]

        for a in self.arglist_state:
            expr_before = symbolic_model.state_model[a]
            expr_after = expr_before.subs(subs_set)
            yield a, expr_after

    def _translate_return(self):
        content = ", ".join(
            (".{name}={name}".format(name=name) for name in self.arglist_state)
        )
        return "State{" + content + "}"

    def ccode_model(self):
        return "{impl}\nreturn {return_};".format(
            impl=self._impl.compile(),
            return_=self._return,
        )

    def state_members(self):
        return "\n".join(
            "double {name};".format(name=name) for name in self.arglist_state
        )

    def control_members(self):
        return "\n".join(
            "double {name};".format(name=name) for name in self.arglist_control
        )


def _generate_function_bodies(header_location, symbolic_model, config):
    generator = Model(symbolic_model, config)

    # For .../generated/formak/xyz.h
    # I want formak/xyz.h , so strip a leading generated prefix if present
    assert "generated/" in header_location
    header_include = header_location.split("generated/")[-1]

    return {
        "header_include": header_include,
        "Model_model_body": generator.ccode_model(),
        "State_members": generator.state_members(),
        "Control_members": generator.control_members(),
    }


def compile(symbolic_model, *, config=None):
    if config is None:
        config = Config()
    elif isinstance(config, dict):
        config = Config(**config)

    parser = argparse.ArgumentParser(prog="generator.py")
    parser.add_argument("--templates")
    parser.add_argument("--header")
    parser.add_argument("--source")

    args = parser.parse_args()

    header_template, source_template = args.templates.split(" ")

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

    inserts = _generate_function_bodies(args.header, symbolic_model, config)

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

def compile_ekf(
    state_model, process_noise, sensor_models, sensor_noises, *, config=None
):
    if config is None:
        config = Config()
    elif isinstance(config, dict):
        config = Config(**config)

    parser = argparse.ArgumentParser(prog="generator.py")
    parser.add_argument("--templates")
    parser.add_argument("--header")
    parser.add_argument("--source")

    args = parser.parse_args()

    header_template, source_template = args.templates.split(" ")

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

    inserts = _generate_ekf_function_bodies(args.header, symbolic_model, config)

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
