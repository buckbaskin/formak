import argparse
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from itertools import product
from os import scandir, walk
from os.path import dirname
from typing import Any, List, Tuple

import jinja2
from formak.exceptions import ModelConstructionError
from jinja2 import Environment, FileSystemLoader, select_autoescape
from jinja2.exceptions import TemplateNotFound
from sympy import Symbol, ccode, diff

DEFAULT_MODULES = ("scipy", "numpy", "math")


@dataclass
class Config:
    common_subexpression_elimination: bool = True
    python_modules: List[str] = DEFAULT_MODULES


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
        for lvalue, expr in self._statements:
            cc_expr = ccode(expr)
            yield f"{lvalue} = {cc_expr};"


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
                Symbol("input_state.{}()".format(member)),
            )
            for member in self.arglist_state
        ] + [
            (
                member,
                Symbol("input_control.{}()".format(member)),
            )
            for member in self.arglist_control
        ]

        for a in self.arglist_state:
            expr_before = symbolic_model.state_model[a]
            expr_after = expr_before.subs(subs_set)
            yield f"double {a.name}", expr_after

    def _translate_return(self):
        content = ", ".join(str(symbol) for symbol in self.arglist_state)
        return "State({" + content + "})"

    def model_body(self):
        return "{impl}\nreturn {return_};".format(
            impl=self._impl.compile(),
            return_=self._return,
        )

    def state_members(self):
        indent = " " * 4
        return f"\n{indent}".join(
            "double& %s() { return data(%s, 0); }\n%sdouble %s() const { return data(%s, 0); }"
            % (symbol.name, idx, indent, symbol.name, idx)
            for idx, symbol in enumerate(self.arglist_state)
        )

    def control_members(self):
        return "\n".join(
            "double& %s() {return data(%s, 0); }\ndouble %s() const {return data(%s, 0); }"
            % (name, idx, name, idx)
            for idx, name in enumerate(self.arglist_control)
        )

    def state_options_constructor_initializer_list(self):
        return (
            "data(" + ", ".join(f"options.{name}" for name in self.arglist_state) + ")"
        )

    def control_options_constructor_initializer_list(self):
        return (
            "data("
            + ", ".join(f"options.{name}" for name in self.arglist_control)
            + ")"
        )

    def stateoptions_members(self):
        indent = " " * 4
        return f"\n{indent}".join(
            f"double {symbol.name} = 0.0;" for symbol in self.arglist_state
        )

    def controloptions_members(self):
        return "\n".join(
            f"double {symbol.name} = 0.0;" for symbol in self.arglist_control
        )


# size is the size of the reading for the EKF, not the size of the type
ReadingT = namedtuple(
    "ReadingT",
    [
        "typename",
        "size",
        "identifier",
        "members",
        "initializer_list",
        "Options_members",
        "SensorModel_model_body",
        "SensorModel_covariance_body",
        "SensorModel_jacobian_body",
    ],
)


class ExtendedKalmanFilter:
    """C++ implementation of the EKF."""

    def __init__(
        self, state_model, process_noise, sensor_models, sensor_noises, config
    ):
        if isinstance(config, dict):
            config = Config(**config)
        assert isinstance(config, Config)
        assert isinstance(process_noise, dict)

        # TODO(buck): This is lots of duplication with the model
        self.state_size = len(state_model.state)
        self.control_size = len(state_model.control)

        self.arglist_state = sorted(list(state_model.state), key=lambda x: x.name)
        self.arglist_control = sorted(list(state_model.control), key=lambda x: x.name)
        self.arglist = [state_model.dt] + self.arglist_state + self.arglist_control

        self._process_model = BasicBlock(self._translate_process_model(state_model))
        self._process_jacobian = BasicBlock(
            self._translate_process_jacobian(state_model)
        )

        self._control_jacobian = BasicBlock(
            self._translate_control_jacobian(state_model)
        )
        self._control_covariance = BasicBlock(
            self._translate_control_covariance(process_noise)
        )

        # TODO(buck): Translate the sensor models dictionary contents into BasicBlocks
        self.sensorlist = sorted(
            [(k, v, sensor_noises[k]) for k, v in sensor_models.items()]
        )

        self._return = self._translate_return()

    def _translate_process_model(self, symbolic_model):
        subs_set = [
            (
                member,
                Symbol("input.state.{}()".format(member)),
            )
            for member in self.arglist_state
        ] + [
            (
                member,
                Symbol("input_control.{}()".format(member)),
            )
            for member in self.arglist_control
        ]

        for a in self.arglist_state:
            expr_before = symbolic_model.state_model[a]
            expr_after = expr_before.subs(subs_set)
            yield f"double {a.name}", expr_after

    def process_model_body(self):
        return "{impl}\nreturn {return_};".format(
            impl=self._process_model.compile(),
            return_=self._return,
        )

    def _translate_process_jacobian(self, symbolic_model):
        subs_set = [
            (
                member,
                Symbol("input.state.{}()".format(member)),
            )
            for member in self.arglist_state
        ] + [
            (
                member,
                Symbol("input_control.{}()".format(member)),
            )
            for member in self.arglist_control
        ]

        for idx, symbol in enumerate(self.arglist_state):
            model = symbolic_model.state_model[symbol]
            for state_idx, state in enumerate(self.arglist_state):
                assignment = f"jacobian({idx}, {state_idx})"
                expr_before = diff(model, state)
                expr_after = expr_before.subs(subs_set)
                yield assignment, expr_after

    def process_jacobian_body(self):
        typename = "ExtendedKalmanFilter"
        prefix = f"{typename}::ProcessJacobianT jacobian;"
        impl = self._process_jacobian.compile()
        return f"{prefix}\n{impl}\nreturn jacobian;"

    def _translate_control_jacobian(self, symbolic_model):
        subs_set = [
            (
                member,
                Symbol("input.state.{}()".format(member)),
            )
            for member in self.arglist_state
        ] + [
            (
                member,
                Symbol("input_control.{}()".format(member)),
            )
            for member in self.arglist_control
        ]

        for idx, symbol in enumerate(self.arglist_state):
            model = symbolic_model.state_model[symbol]
            for control_idx, control in enumerate(self.arglist_control):
                assignment = f"jacobian({idx}, {control_idx})"
                expr_before = diff(model, control)
                expr_after = expr_before.subs(subs_set)
                yield assignment, expr_after

    def control_jacobian_body(self):
        typename = "ExtendedKalmanFilter"
        prefix = f"{typename}::ControlJacobianT jacobian;"
        impl = self._control_jacobian.compile()
        return f"{prefix}\n{impl}\nreturn jacobian;"

    def _translate_return(self):
        content = ", ".join(
            ".{name}={name}".format(name=name) for name in self.arglist_state
        )
        return "State({" + content + "});"

    def sensorid_members(self, verbose=True):
        # TODO(buck): Add a verbose flag option that will print out the generated class members
        # TODO(buck): remove the default True in favor of the flag option
        enum_names = [
            "{name}".format(name=name.upper()) for name, _, _ in self.sensorlist
        ]
        if verbose:
            print(f"sensorid_members: enum_names: {enum_names}")
        return ",\n".join(enum_names)

    def state_members(self):
        indent = " " * 4
        return f"\n{indent}".join(
            "double& %s() { return data(%s, 0); }\n%sdouble %s() const { return data(%s, 0); }"
            % (symbol.name, idx, indent, symbol.name, idx)
            for idx, symbol in enumerate(self.arglist_state)
        )

    def stateoptions_members(self):
        return "\n".join(
            f"double {symbol.name} = 0.0;" for symbol in self.arglist_state
        )

    def controloptions_members(self):
        return "\n".join(
            f"double {symbol.name} = 0.0;" for symbol in self.arglist_control
        )

    def state_options_constructor_initializer_list(self):
        return (
            "data(" + ", ".join(f"options.{name}" for name in self.arglist_state) + ")"
        )

    def control_options_constructor_initializer_list(self):
        return (
            "data("
            + ", ".join(f"options.{name}" for name in self.arglist_control)
            + ")"
        )

    def covariance_members(self):
        # TODO(buck): Need to add covariance terms
        return "\n".join(
            "double& %s() { return data(%s, %s); }\ndouble %s() const { return data(%s, %s); }"
            % (symbol.name, idx, idx, symbol.name, idx, idx)
            for idx, symbol in enumerate(self.arglist_state)
        )

    def control_members(self):
        return "\n".join(
            "double& %s() {return data(%s, 0); }\ndouble %s() const {return data(%s, 0); }"
            % (name, idx, name, idx)
            for idx, name in enumerate(self.arglist_control)
        )

    def _translate_control_covariance(self, covariance):
        for i, iKey in enumerate(self.arglist_control):
            for j, jKey in enumerate(self.arglist_control):
                if (iKey, jKey) in covariance:
                    value = covariance[(iKey, jKey)]
                elif (jKey, iKey) in covariance:
                    value = covariance[(jKey, iKey)]
                elif i == j and iKey in covariance:
                    value = covariance[iKey]
                else:
                    value = 0.0
                yield f"covariance({i}, {j})", value
                yield f"covariance({j}, {i})", value

    def control_covariance_body(self):
        typename = "ExtendedKalmanFilter"
        prefix = f"{typename}::CovarianceT covariance;"
        body = self._control_covariance
        suffix = "return covariance;"
        return prefix + "\n" + body.compile() + "\n" + suffix

    def _translate_sensor_model(self, sensor_model_mapping):
        subs_set = [
            (
                member,
                Symbol("input.state.{}()".format(member)),
            )
            for member in self.arglist_state
        ]
        for predicted_reading, model in sorted(list(sensor_model_mapping.items())):
            assignment = str(predicted_reading)
            expr_before = model
            expr_after = expr_before.subs(subs_set)
            yield f"double {assignment}", expr_after

    def reading_types(self, verbose=True):
        for name, sensor_model_mapping, sensor_noise in self.sensorlist:
            typename = name.title()
            identifier = f"SensorId::{name.upper()}"
            members = "\n".join(
                "double& %s() { return data(%d, 0); }" % (name, idx)
                for idx, name in enumerate(sorted(list(sensor_model_mapping.keys())))
            )
            size = len(sensor_model_mapping)
            if verbose:
                print(
                    f"reading_types: name: {name} reading_type: {typename} {identifier} members:\n{members}"
                )
                print("Model:")
                for predicted_reading, model in sorted(
                    list(sensor_model_mapping.items())
                ):
                    print(f"Modeling {predicted_reading} as function of state: {model}")

            body = BasicBlock(self._translate_sensor_model(sensor_model_mapping))
            return_ = (
                "return {}Options{{".format(typename)
                + ", ".join(
                    str(reading)
                    for reading in sorted(list(sensor_model_mapping.keys()))
                )
                + "};"
            )
            # TODO(buck): Move this line handling to the template?
            SensorModel_model_body = body.compile() + "\n" + return_
            SensorModel_covariance_body = self._translate_sensor_covariance(
                typename, sensor_noise
            )
            SensorModel_jacobian_body = self._translate_sensor_jacobian(
                typename, sensor_model_mapping
            )

            initializer_list = (
                "data("
                + ", ".join(
                    f"options.{name}"
                    for name in sorted(list(sensor_model_mapping.keys()))
                )
                + ")"
            )
            Options_members = "\n".join(
                f"double {str(symbol)} = 0.0;"
                for symbol in sorted(list(sensor_model_mapping.keys()))
            )

            yield ReadingT(
                identifier=identifier,
                initializer_list=initializer_list,
                members=members,
                Options_members=Options_members,
                SensorModel_covariance_body=SensorModel_covariance_body,
                SensorModel_jacobian_body=SensorModel_jacobian_body,
                SensorModel_model_body=SensorModel_model_body,
                size=size,
                typename=typename,
            )

    def _translate_sensor_jacobian_impl(self, sensor_model_mapping):
        subs_set = [
            (
                member,
                Symbol("input.state.{}()".format(member)),
            )
            for member in self.arglist_state
        ]

        for reading_idx, (_predicted_reading, model) in enumerate(
            sorted(list(sensor_model_mapping.items()))
        ):
            for state_idx, state in enumerate(self.arglist_state):
                assignment = f"jacobian({reading_idx}, {state_idx})"
                expr_before = diff(model, state)
                expr_after = expr_before.subs(subs_set)
                yield assignment, expr_after

    def _translate_sensor_jacobian(self, typename, sensor_model_mapping):
        prefix = f"{typename}::SensorJacobianT jacobian;"
        body = BasicBlock(self._translate_sensor_jacobian_impl(sensor_model_mapping))
        suffix = "return jacobian;"
        return prefix + "\n" + body.compile() + "\n" + suffix

    def _translate_sensor_covariance_impl(self, covariance):
        rows, cols = covariance.shape
        for i in range(rows):
            for j in range(cols):
                yield f"covariance({i}, {j})", covariance[i, j]

    def _translate_sensor_covariance(self, typename, covariance):
        prefix = f"{typename}::CovarianceT covariance;"
        body = BasicBlock(self._translate_sensor_covariance_impl(covariance))
        suffix = "return covariance;"
        return prefix + "\n" + body.compile() + "\n" + suffix


def _generate_model_function_bodies(header_location, namespace, symbolic_model, config):
    generator = Model(symbolic_model, config)

    # For .../generated/formak/xyz.h
    # I want formak/xyz.h , so strip a leading generated prefix if present
    assert "generated/" in header_location
    header_include = header_location.split("generated/")[-1]

    return {
        "Control_members": generator.control_members(),
        "Control_options_constructor_initializer_list": generator.control_options_constructor_initializer_list(),
        "Control_size": generator.control_size,
        "ControlOptions_members": generator.controloptions_members(),
        "header_include": header_include,
        "Model_model_body": generator.model_body(),
        "namespace": namespace,
        "State_members": generator.state_members(),
        "State_options_constructor_initializer_list": generator.state_options_constructor_initializer_list(),
        "State_size": generator.state_size,
        "StateOptions_members": generator.stateoptions_members(),
    }


def _generate_ekf_function_bodies(
    header_location,
    namespace,
    state_model,
    process_noise,
    sensor_models,
    sensor_noises,
    config,
):
    generator = ExtendedKalmanFilter(
        state_model, process_noise, sensor_models, sensor_noises, config
    )

    # For .../generated/formak/xyz.h
    # I want formak/xyz.h , so strip a leading generated prefix if present
    assert "generated/" in header_location
    header_include = header_location.split("generated/")[-1]

    # TODO(buck): Eventually should split out the code generation for the header and the source
    return {
        "Control_members": generator.control_members(),
        "Control_options_constructor_initializer_list": generator.control_options_constructor_initializer_list(),
        "Control_size": generator.control_size,
        "ControlOptions_members": generator.controloptions_members(),
        "Covariance_members": generator.covariance_members(),
        "ExtendedKalmanFilterProcessModel_model_body": generator.process_model_body(),
        "ExtendedKalmanFilterProcessModel_process_jacobian_body": generator.process_jacobian_body(),
        "ExtendedKalmanFilterProcessModel_control_jacobian_body": generator.control_jacobian_body(),
        "ExtendedKalmanFilterProcessModel_covariance_body": generator.control_covariance_body(),
        "header_include": header_include,
        "namespace": namespace,
        "reading_types": list(generator.reading_types()),
        "SensorId_members": generator.sensorid_members(),
        "State_members": generator.state_members(),
        "State_options_constructor_initializer_list": generator.state_options_constructor_initializer_list(),
        "State_size": generator.state_size,
        "StateOptions_members": generator.stateoptions_members(),
    }


def _parse_raw_templates(arg, verbose=True):
    raw_templates = arg.split(" ")
    templates = defaultdict(dict)

    EXPECT_PREFIX = "py/formak/templates/"

    for template_str in raw_templates:
        if verbose:
            print(f"Examining template: {template_str}")
        if not template_str.startswith(EXPECT_PREFIX):
            raise ValueError(
                f"Template {template_str} did not start with expected prefix {EXPECT_PREFIX}"
            )

        if template_str.endswith(".cpp"):
            templates[template_str[len(EXPECT_PREFIX) : -4]][".cpp"] = template_str
        elif template_str.endswith(".h"):
            templates[template_str[len(EXPECT_PREFIX) : -2]][".h"] = template_str
        elif template_str.endswith(".hpp"):
            templates[template_str[len(EXPECT_PREFIX) : -4]][".hpp"] = template_str
        else:
            raise ValueError(
                f"Template {template_str} did not end with expected suffix"
            )

    return templates


def compile(symbolic_model, *, config=None):
    if config is None:
        config = Config()
    elif isinstance(config, dict):
        config = Config(**config)

    parser = argparse.ArgumentParser(prog="generator.py")
    parser.add_argument("--templates")
    parser.add_argument("--header")
    parser.add_argument("--source")
    parser.add_argument("--namespace")

    args = parser.parse_args()

    templates = _parse_raw_templates(args.templates)

    # TODO(buck): This '.h', '.cpp' feels like it should be a little dataclass
    # TODO(buck): Then it becomes templates['model']['.h'] -> templates['model'].h
    # TODO(buck): Which feels like a little bit of a coding pun
    header_template = templates["formak_model"][".h"]
    source_template = templates["formak_model"][".cpp"]

    templates_base_path = dirname(header_template)
    assert templates_base_path == dirname(source_template)

    env = Environment(
        loader=FileSystemLoader(templates_base_path),
        autoescape=select_autoescape(),
        undefined=jinja2.StrictUndefined,
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

    inserts = _generate_model_function_bodies(
        args.header, args.namespace, symbolic_model, config
    )

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

    assert isinstance(process_noise, dict)
    allowed_keys = set(
        list(state_model.control)
        + [
            (x, y)
            for x, y in product(state_model.control, state_model.control)
            if x != y
        ]
    )
    for key in process_noise:
        if key not in allowed_keys:
            raise ModelConstructionError(f"Key {key} not in allowlist {allowed_keys}")

    parser = argparse.ArgumentParser(prog="generator.py")
    parser.add_argument("--templates")
    parser.add_argument("--header")
    parser.add_argument("--source")
    parser.add_argument("--namespace")

    args = parser.parse_args()

    templates = _parse_raw_templates(args.templates)

    header_template = templates["formak_ekf"][".hpp"]
    source_template = templates["formak_ekf"][".cpp"]

    # TODO(buck): This won't scale well to organizing templates in folders
    templates_base_path = dirname(header_template)
    assert templates_base_path == dirname(source_template)

    env = Environment(
        loader=FileSystemLoader(templates_base_path), autoescape=select_autoescape()
    )

    try:
        header_template = env.get_template("formak_ekf.hpp")
        source_template = env.get_template("formak_ekf.cpp")
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

    inserts = _generate_ekf_function_bodies(
        args.header,
        args.namespace,
        state_model,
        process_noise,
        sensor_models,
        sensor_noises,
        config,
    )

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
