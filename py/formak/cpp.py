import argparse
import logging
from collections import namedtuple
from dataclasses import dataclass
from itertools import chain, count
from typing import Any, List, Tuple

from formak.ast_tools import (
    Arg,
    ClassDef,
    CompileState,
    ConstructorDeclaration,
    ConstructorDefinition,
    EnumClassDef,
    Escape,
    ForwardClassDeclaration,
    FromFileTemplate,
    FunctionDeclaration,
    FunctionDef,
    HeaderFile,
    MemberDeclaration,
    Namespace,
    Return,
    SourceFile,
    Templated,
    UsingDeclaration,
)
from formak.exceptions import ModelConstructionError
from sympy import Symbol, ccode, cse, diff

from formak import common

DEFAULT_MODULES = ("scipy", "numpy", "math")

logger = logging.getLogger(__name__)


@dataclass
class Config:
    common_subexpression_elimination: bool = True
    python_modules: List[str] = DEFAULT_MODULES
    extra_validation: bool = False


@dataclass
class CppCompileResult:
    success: bool
    header_path: str = None
    source_path: str = None


class BasicBlock:
    def __init__(self, statements: List[Tuple[str, Any]], indent=0, *, config):
        # should be Tuple[str, sympy expression]
        statements = list(statements)
        self._targets = [k for k, _ in statements]
        self._exprs = [v for _, v in statements]
        self._indent = indent
        self._config = config

    def compile(self):
        result = "\n".join(self._compile_impl())
        logger.debug(f"Compile Result: {result}")
        return result

    def _compile_impl(self):
        prefix = []
        body = self._exprs

        if self._config.common_subexpression_elimination:
            prefix, body = cse(body, symbols=(Symbol(f"_t{i}") for i in count()))

        # Note: The list of statements is ordered and can get CSE or reordered within the block because we know it is straight calculation without control flow (a basic block)
        for target, expr in prefix:
            assert isinstance(target, Symbol)
            cc_expr = ccode(expr)
            yield f"{' ' * self._indent}double {target} = {cc_expr};"

        for target, expr in zip(self._targets, body):
            cc_expr = ccode(expr)
            yield f"{' ' * self._indent}{target} = {cc_expr};"


class Model:
    """C++ implementation of the model."""

    def __init__(
        self, symbolic_model, calibration_map, namespace, header_include, config
    ):
        # TODO(buck): Enable mypy for type checking
        # TODO(buck): Move all type assertions to either __init__ (constructor) or mypy?
        # assert isinstance(symbolic_model, ui.Model)
        if isinstance(config, dict):
            config = Config(**config)
        assert isinstance(config, Config)

        self.enable_EKF = False
        self.namespace = namespace
        self.header_include = header_include

        self.sensorlist = {}

        self.state_size = len(symbolic_model.state)
        self.control_size = len(symbolic_model.control)
        self.calibration_size = len(symbolic_model.calibration)

        self.arglist_state = sorted(list(symbolic_model.state), key=lambda x: x.name)
        self.arglist_calibration = sorted(
            list(symbolic_model.calibration), key=lambda x: x.name
        )
        self.arglist_control = sorted(
            list(symbolic_model.control), key=lambda x: x.name
        )
        self.arglist = (
            [symbolic_model.dt]
            + self.arglist_state
            + self.arglist_calibration
            + self.arglist_control
        )

        if self.calibration_size > 0:
            if len(calibration_map) == 0:
                map_lite = "\n  , ".join(
                    [f"{k}: ..." for k in self.arglist_calibration[:3]]
                )
                if len(self.arglist_calibration) > 3:
                    map_lite += ", ..."
                raise ModelConstructionError(
                    f"Model with empty specification of calibration_map:\n{{{map_lite}}}"
                )
            if len(calibration_map) != self.calibration_size:
                missing_from_map = set(symbolic_model.calibration) - set(
                    calibration_map.keys()
                )
                extra_from_map = set(calibration_map.keys()) - set(
                    symbolic_model.calibration
                )
                missing = ""
                if len(missing_from_map) > 0:
                    missing = f"\nMissing: {missing_from_map}"
                extra = ""
                if len(extra_from_map) > 0:
                    extra = f"\nExtra: {extra_from_map}"
                raise ModelConstructionError(f"Mismatched Calibration:{missing}{extra}")

        self._model = BasicBlock(
            self._translate_model(symbolic_model), indent=4, config=config
        )

        self._return = self._translate_return()

    def _translate_model(self, symbolic_model):
        subs_set = (
            [
                (
                    member,
                    Symbol("input_state.{}()".format(member)),
                )
                for member in self.arglist_state
            ]
            + [
                (
                    member,
                    Symbol("input_calibration.{}()".format(member)),
                )
                for member in self.arglist_calibration
            ]
            + [
                (
                    member,
                    Symbol("input_control.{}()".format(member)),
                )
                for member in self.arglist_control
            ]
        )

        for a in self.arglist_state:
            expr_before = symbolic_model.state_model[a]
            expr_after = expr_before.subs(subs_set)
            yield f"double {a.name}", expr_after

    def _translate_return(self):
        content = ", ".join(str(symbol) for symbol in self.arglist_state)
        return "State({" + content + "})"

    def model_body(self):
        indent = " " * 4
        return "{impl}\n{indent}return {return_};".format(
            impl=self._model.compile(),
            indent=indent,
            return_=self._return,
        )

    def enable_control(self):
        return self.control_size > 0

    def enable_calibration(self):
        return self.calibration_size > 0


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
        "sensor_model_mapping",
    ],
)


class ExtendedKalmanFilter:
    """C++ implementation of the EKF."""

    def __init__(
        self,
        state_model,
        process_noise,
        sensor_models,
        sensor_noises,
        namespace,
        header_include,
        config,
        calibration_map=None,
    ):
        if isinstance(config, dict):
            config = Config(**config)
        assert isinstance(config, Config)
        assert isinstance(process_noise, dict)

        self.enable_EKF = True
        self.config = config
        self.namespace = namespace
        self.header_include = header_include

        # TODO(buck): This is lots of duplication with the model
        self.state_size = len(state_model.state)
        self.calibration_size = len(state_model.calibration)
        self.control_size = len(state_model.control)

        self.arglist_state = sorted(list(state_model.state), key=lambda x: x.name)
        self.arglist_calibration = sorted(
            list(state_model.calibration), key=lambda x: x.name
        )
        self.arglist_control = sorted(list(state_model.control), key=lambda x: x.name)
        self.arglist = (
            [state_model.dt]
            + self.arglist_state
            + self.arglist_calibration
            + self.arglist_control
        )

        if self.calibration_size > 0:
            if len(calibration_map) == 0:
                map_lite = "\n  , ".join(
                    [f"{k}: ..." for k in self.arglist_calibration[:3]]
                )
                if len(self.arglist_calibration) > 3:
                    map_lite += ", ..."
                raise ModelConstructionError(
                    f"Model Missing specification of calibration_map:\n{{{map_lite}}}"
                )
            if len(calibration_map) != self.calibration_size:
                missing_from_map = set(state_model.calibration) - set(
                    calibration_map.keys()
                )
                extra_from_map = set(calibration_map.keys()) - set(
                    state_model.calibration
                )
                missing = ""
                if len(missing_from_map) > 0:
                    missing = f"\nMissing: {missing_from_map}"
                extra = ""
                if len(extra_from_map) > 0:
                    extra = f"\nExtra: {extra_from_map}"
                raise ModelConstructionError(f"Mismatched Calibration:{missing}{extra}")

        self._process_model = BasicBlock(
            self._translate_process_model(state_model),
            indent=4,
            config=config,
        )
        self._process_jacobian = BasicBlock(
            self._translate_process_jacobian(state_model),
            indent=4,
            config=config,
        )

        self._control_jacobian = BasicBlock(
            self._translate_control_jacobian(state_model),
            indent=4,
            config=config,
        )
        self._control_covariance = BasicBlock(
            self._translate_control_covariance(process_noise),
            indent=4,
            config=config,
        )

        # TODO(buck): Translate the sensor models dictionary contents into BasicBlocks
        self.sensorlist = sorted(
            [(k, v, sensor_noises[k]) for k, v in sensor_models.items()]
        )

        self._return = self._translate_return()

    def _translate_process_model(self, symbolic_model):
        subs_set = (
            [
                (
                    member,
                    Symbol("input.state.{}()".format(member)),
                )
                for member in self.arglist_state
            ]
            + [
                (
                    member,
                    Symbol("input_calibration.{}()".format(member)),
                )
                for member in self.arglist_calibration
            ]
            + [
                (
                    member,
                    Symbol("input_control.{}()".format(member)),
                )
                for member in self.arglist_control
            ]
        )

        for a in self.arglist_state:
            expr_before = symbolic_model.state_model[a]
            expr_after = expr_before.subs(subs_set)
            yield f"double {a.name}", expr_after

    def process_model_body(self):
        indent = " " * 4
        return "{impl}\n{indent}return {return_};".format(
            impl=self._process_model.compile(),
            indent=indent,
            return_=self._return,
        )

    def _translate_process_jacobian(self, symbolic_model):
        subs_set = (
            [
                (
                    member,
                    Symbol("input.state.{}()".format(member)),
                )
                for member in self.arglist_state
            ]
            + [
                (
                    member,
                    Symbol("input_calibration.{}()".format(member)),
                )
                for member in self.arglist_calibration
            ]
            + [
                (
                    member,
                    Symbol("input_control.{}()".format(member)),
                )
                for member in self.arglist_control
            ]
        )

        for idx, symbol in enumerate(self.arglist_state):
            model = symbolic_model.state_model[symbol]
            for state_idx, state in enumerate(self.arglist_state):
                assignment = f"jacobian({idx}, {state_idx})"
                expr_before = diff(model, state)
                expr_after = expr_before.subs(subs_set)
                yield assignment, expr_after

    def process_jacobian_body(self):
        indent = " " * 4
        typename = "ExtendedKalmanFilter"
        prefix = f"{indent}{typename}::ProcessJacobianT jacobian;"
        impl = self._process_jacobian.compile()
        return f"{prefix}\n{impl}\n{indent}return jacobian;"

    def _translate_control_jacobian(self, symbolic_model):
        subs_set = (
            [
                (
                    member,
                    Symbol("input.state.{}()".format(member)),
                )
                for member in self.arglist_state
            ]
            + [
                (
                    member,
                    Symbol("input_calibration.{}()".format(member)),
                )
                for member in self.arglist_calibration
            ]
            + [
                (
                    member,
                    Symbol("input_control.{}()".format(member)),
                )
                for member in self.arglist_control
            ]
        )

        for idx, symbol in enumerate(self.arglist_state):
            model = symbolic_model.state_model[symbol]
            for control_idx, control in enumerate(self.arglist_control):
                assignment = f"jacobian({idx}, {control_idx})"
                expr_before = diff(model, control)
                expr_after = expr_before.subs(subs_set)
                yield assignment, expr_after

    def control_jacobian_body(self):
        indent = " " * 4
        typename = "ExtendedKalmanFilter"
        prefix = f"{indent}{typename}::ControlJacobianT jacobian;"
        impl = self._control_jacobian.compile()
        return f"{prefix}\n{impl}\n{indent}return jacobian;"

    def _translate_return(self):
        content = ", ".join(
            ".{name}={name}".format(name=name) for name in self.arglist_state
        )
        return "State({" + content + "});"

    def enable_control(self):
        return self.control_size > 0

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
                if i != j:
                    yield f"covariance({j}, {i})", value

    def control_covariance_body(self):
        indent = " " * 4
        typename = "ExtendedKalmanFilter"
        prefix = f"{indent}{typename}::CovarianceT covariance;"
        body = self._control_covariance
        suffix = f"{indent}return covariance;"
        return prefix + "\n" + body.compile() + "\n" + suffix

    def enable_calibration(self):
        return self.calibration_size > 0

    def _translate_sensor_model(self, sensor_model_mapping):
        subs_set = [
            (
                member,
                Symbol("input.state.{}()".format(member)),
            )
            for member in self.arglist_state
        ] + [
            (
                member,
                Symbol("input_calibration.{}()".format(member)),
            )
            for member in self.arglist_calibration
        ]
        for predicted_reading, model in sorted(list(sensor_model_mapping.items())):
            assignment = str(predicted_reading)
            expr_before = model
            expr_after = expr_before.subs(subs_set)
            yield f"double {assignment}", expr_after

    def reading_types(self, verbose=False):
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

            body = BasicBlock(
                self._translate_sensor_model(sensor_model_mapping),
                indent=4,
                config=self.config,
            )
            indent = " " * 4
            return_ = (
                "{}return {}Options{{".format(indent, typename)
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
                sensor_model_mapping=sensor_model_mapping,
            )

    def _translate_sensor_jacobian_impl(self, sensor_model_mapping):
        subs_set = [
            (
                member,
                Symbol("input.state.{}()".format(member)),
            )
            for member in self.arglist_state
        ] + [
            (
                member,
                Symbol("input_calibration.{}()".format(member)),
            )
            for member in self.arglist_calibration
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
        indent = " " * 4
        prefix = f"{indent}{typename}::SensorJacobianT jacobian;"
        body = BasicBlock(
            self._translate_sensor_jacobian_impl(sensor_model_mapping),
            indent=4,
            config=self.config,
        )
        suffix = f"{indent}return jacobian;"
        return prefix + "\n" + body.compile() + "\n" + suffix

    def _translate_sensor_covariance_impl(self, covariance):
        rows, cols = covariance.shape
        for i in range(rows):
            for j in range(cols):
                yield f"covariance({i}, {j})", covariance[i, j]

    def _translate_sensor_covariance(self, typename, covariance):
        indent = " " * 4
        prefix = f"{indent}{typename}::CovarianceT covariance;"
        body = BasicBlock(
            self._translate_sensor_covariance_impl(covariance),
            indent=4,
            config=self.config,
        )
        suffix = f"{indent}return covariance;"
        return prefix + "\n" + body.compile() + "\n" + suffix


def _generate_model_function_bodies(
    header_location, namespace, symbolic_model, calibration_map, config
):

    # For .../generated/formak/xyz.h
    # I want formak/xyz.h , so strip a leading generated prefix if present
    if header_location is not None and "generated/" in header_location:
        header_include = header_location.split("generated/")[-1]
    else:
        header_include = "generated_to_stdout.h"

    generator = Model(
        symbolic_model, calibration_map, namespace, header_include, config
    )

    return generator


def _generate_ekf_function_bodies(
    header_location,
    namespace,
    state_model,
    process_noise,
    sensor_models,
    sensor_noises,
    calibration_map,
    config,
):
    # For .../generated/formak/xyz.h
    # I want formak/xyz.h , so strip a leading generated prefix if present
    if header_location is not None and "generated/" in header_location:
        header_include = header_location.split("generated/")[-1]
    else:
        header_include = "generated_to_stdout.h"

    generator = ExtendedKalmanFilter(
        state_model=state_model,
        process_noise=process_noise,
        sensor_models=sensor_models,
        sensor_noises=sensor_noises,
        calibration_map=calibration_map,
        namespace=namespace,
        header_include=header_include,
        config=config,
    )

    return generator


def _compile_argparse():
    parser = argparse.ArgumentParser(prog="generator.py")
    parser.add_argument("--header")
    parser.add_argument("--source")
    parser.add_argument("--namespace")

    args = parser.parse_args()
    return args


def header_from_ast(*, generator):
    StateOptions = ClassDef(
        "struct",
        "StateOptions",
        bases=[],
        body=[
            MemberDeclaration("double", member, 0.0)
            for member in generator.arglist_state
        ],
    )
    State = ClassDef(
        "struct",
        "State",
        bases=[],
        body=[
            MemberDeclaration("static constexpr size_t", "rows", generator.state_size),
            MemberDeclaration("static constexpr size_t", "cols", 1),
            UsingDeclaration("DataT", "Eigen::Matrix<double, rows, cols>"),
            ConstructorDeclaration(),  # No args constructor gets default constructor
            ConstructorDeclaration(args=[Arg("const StateOptions&", "options")]),
        ]
        + list(
            chain.from_iterable(
                [
                    (
                        FunctionDef(
                            "double&",
                            name,
                            args=[],
                            modifier="",
                            body=[
                                Return(f"data({idx}, 0)"),
                            ],
                        ),
                        FunctionDef(
                            "double",
                            name,
                            args=[],
                            modifier="const",
                            body=[
                                Return(f"data({idx}, 0)"),
                            ],
                        ),
                    )
                    for idx, name in enumerate(generator.arglist_state)
                ]
            )
        )
        + [
            MemberDeclaration("DataT", "data", "DataT::Zero()"),
        ],
    )
    body = [
        StateOptions,
        State,
    ]

    if generator.enable_control():
        ControlOptions = ClassDef(
            "struct",
            "ControlOptions",
            bases=[],
            body=[
                MemberDeclaration("double", member, 0.0)
                for member in generator.arglist_control
            ],
        )
        Control = ClassDef(
            "struct",
            "Control",
            bases=[],
            body=[
                MemberDeclaration(
                    "static constexpr size_t", "rows", generator.control_size
                ),
                MemberDeclaration("static constexpr size_t", "cols", 1),
                UsingDeclaration("DataT", "Eigen::Matrix<double, rows, cols>"),
                ConstructorDeclaration(),  # No args constructor gets default constructor
                ConstructorDeclaration(args=[Arg("const ControlOptions&", "options")]),
            ]
            + list(
                chain.from_iterable(
                    [
                        (
                            FunctionDef(
                                "double&",
                                name,
                                args=[],
                                modifier="",
                                body=[
                                    Return(f"data({idx}, 0)"),
                                ],
                            ),
                            FunctionDef(
                                "double",
                                name,
                                args=[],
                                modifier="const",
                                body=[
                                    Return(f"data({idx}, 0)"),
                                ],
                            ),
                        )
                        for idx, name in enumerate(generator.arglist_control)
                    ]
                )
            )
            + [
                MemberDeclaration("DataT", "data", "DataT::Zero()"),
            ],
        )
        body.append(ControlOptions)
        body.append(Control)
    if generator.enable_calibration():
        CalibrationOptions = ClassDef(
            "struct",
            "CalibrationOptions",
            bases=[],
            body=[
                MemberDeclaration("double", member, 0.0)
                for member in generator.arglist_calibration
            ],
        )
        Calibration = ClassDef(
            "struct",
            "Calibration",
            bases=[],
            body=[
                MemberDeclaration(
                    "static constexpr size_t", "rows", generator.calibration_size
                ),
                MemberDeclaration("static constexpr size_t", "cols", 1),
                UsingDeclaration("DataT", "Eigen::Matrix<double, rows, cols>"),
                ConstructorDeclaration(),  # No args constructor gets default constructor
                ConstructorDeclaration(
                    args=[Arg("const CalibrationOptions&", "options")]
                ),
            ]
            + list(
                chain.from_iterable(
                    [
                        (
                            FunctionDef(
                                "double&",
                                name,
                                args=[],
                                modifier="",
                                body=[
                                    Return(f"data({idx}, 0)"),
                                ],
                            ),
                            FunctionDef(
                                "double",
                                name,
                                args=[],
                                modifier="const",
                                body=[
                                    Return(f"data({idx}, 0)"),
                                ],
                            ),
                        )
                        for idx, name in enumerate(generator.arglist_calibration)
                    ]
                )
            )
            + [
                MemberDeclaration("DataT", "data", "DataT::Zero()"),
            ],
        )
        body.append(CalibrationOptions)
        body.append(Calibration)

    def State_model(*, enable_control, enable_calibration):
        args = [
            Arg("double", "dt"),
            Arg("const State&", "input_state"),
        ]

        if enable_control:
            args.append(Arg("const Control&", "input_control"))
        if enable_calibration:
            args.append(Arg("const Calibration&", "input_calibration"))

        return FunctionDeclaration(
            "State",
            "model",
            args=args,
            modifier="",
        )

    def StateAndVariance_process_model(*, enable_control, enable_calibration):
        args = [
            Arg("double", "dt"),
            Arg("const StateAndVariance&", "input_state"),
        ]

        if enable_calibration:
            args.append(Arg("const Calibration&", "input_calibration"))
        if enable_control:
            args.append(Arg("const Control&", "input_control"))

        return FunctionDeclaration(
            "StateAndVariance",
            "process_model",
            args=args,
            modifier="",
        )

    def StateAndVariance_sensor_model(*, enable_control, enable_calibration):
        args = [
            Arg("const StateAndVariance&", "input"),
        ]

        if enable_calibration:
            args.append(Arg("const Calibration&", "input_calibration"))

        args.append(Arg("const ReadingT&", "input_reading"))

        return Templated(
            [Arg("typename", "ReadingT")],
            FunctionDef(
                "StateAndVariance",
                "sensor_model",
                args=args,
                modifier="",
                body=[
                    FromFileTemplate(
                        "sensor_model.hpp",
                        inserts={
                            "enable_calibration": generator.enable_calibration(),
                        },
                    ),
                ],
            ),
        )

    if generator.enable_EKF:
        Covariance = ClassDef(
            "struct",
            "Covariance",
            bases=[],
            body=[
                MemberDeclaration(
                    "static constexpr size_t", "rows", generator.state_size
                ),
                MemberDeclaration(
                    "static constexpr size_t", "cols", generator.state_size
                ),
                UsingDeclaration("DataT", "Eigen::Matrix<double, rows, cols>"),
            ]
            + list(
                chain.from_iterable(
                    [
                        (
                            FunctionDef(
                                "double&",
                                name,
                                args=[],
                                modifier="",
                                body=[
                                    Return(f"data({idx}, {idx})"),
                                ],
                            ),
                            FunctionDef(
                                "double",
                                name,
                                args=[],
                                modifier="const",
                                body=[
                                    Return(f"data({idx}, {idx})"),
                                ],
                            ),
                        )
                        for idx, name in enumerate(generator.arglist_state)
                    ]
                )
            )
            + [
                MemberDeclaration("DataT", "data", "DataT::Identity()"),
            ],
        )
        body.append(Covariance)
        StateAndVariance = ClassDef(
            "struct",
            "StateAndVariance",
            bases=[],
            body=[
                MemberDeclaration("State", "state"),
                MemberDeclaration("Covariance", "covariance"),
            ],
        )
        body.append(StateAndVariance)
        SensorId = EnumClassDef(
            "SensorId",
            members=[f"{name.upper()}" for name, _, _ in generator.sensorlist],
        )
        body.append(SensorId)
        for reading_type in generator.reading_types():
            body.append(
                ForwardClassDeclaration("struct", f"{reading_type.typename}SensorModel")
            )
            body.append(
                ClassDef(
                    "struct",
                    f"{reading_type.typename}Options",
                    bases=[],
                    body=[
                        MemberDeclaration("double", symbol, 0.0)
                        for symbol in sorted(
                            list(reading_type.sensor_model_mapping.keys())
                        )
                    ],
                )
            )

            standard_args = [Arg("const StateAndVariance&", "input")]
            if generator.enable_calibration():
                standard_args.append(Arg("const Calibration&", "input_calibration"))
            standard_args.append(
                Arg(f"const {reading_type.typename}&", "input_reading")
            )

            body.append(
                ClassDef(
                    "struct",
                    f"{reading_type.typename}",
                    bases=[],
                    body=[
                        UsingDeclaration(
                            "DataT", f"Eigen::Matrix<double, {reading_type.size}, 1>"
                        ),
                        UsingDeclaration(
                            "CovarianceT",
                            f"Eigen::Matrix<double, {reading_type.size}, {reading_type.size}>",
                        ),
                        UsingDeclaration(
                            "InnovationT",
                            f"Eigen::Matrix<double, {reading_type.size}, 1>",
                        ),
                        UsingDeclaration(
                            "KalmanGainT",
                            f"Eigen::Matrix<double, {generator.state_size}, {reading_type.size}>",
                        ),
                        UsingDeclaration(
                            "SensorJacobianT",
                            f"Eigen::Matrix<double, {reading_type.size}, {generator.state_size}>",
                        ),
                        UsingDeclaration(
                            "SensorModel", f"{reading_type.typename}SensorModel"
                        ),
                        ConstructorDeclaration(args=[]),
                        ConstructorDeclaration(
                            args=[
                                Arg(f"const {reading_type.typename}Options&", "options")
                            ]
                        ),
                    ]
                    + [
                        FunctionDef(
                            "double",
                            name,
                            args=[],
                            modifier="",
                            body=[Return(f"data({idx}, 0)")],
                        )
                        for idx, name in enumerate(
                            sorted(list(reading_type.sensor_model_mapping.keys()))
                        )
                    ]
                    + [
                        FunctionDeclaration(
                            f"static {reading_type.typename}",
                            "model",
                            args=standard_args,
                            modifier="",
                        ),
                        FunctionDeclaration(
                            f"static {reading_type.typename}::SensorJacobianT",
                            "jacobian",
                            args=standard_args,
                            modifier="",
                        ),
                        FunctionDeclaration(
                            f"static {reading_type.typename}::CovarianceT",
                            "covariance",
                            args=standard_args,
                            modifier="",
                        ),
                        MemberDeclaration("DataT", "data", "DataT::Zero()"),
                        MemberDeclaration(
                            "constexpr static size_t", "size", reading_type.size
                        ),
                        MemberDeclaration(
                            "constexpr static SensorId",
                            "Identifier",
                            reading_type.identifier,
                        ),
                    ],
                )
            )

            standard_args = [Arg("const StateAndVariance&", "input")]
            if generator.enable_calibration():
                standard_args.append(Arg("const Calibration&", "input_calibration"))
            standard_args.append(
                Arg(f"const {reading_type.typename}&", "input_reading")
            )
            body.append(
                ClassDef(
                    "struct",
                    f"{reading_type.typename}SensorModel",
                    bases=[],
                    body=[
                        FunctionDeclaration(
                            f"static {reading_type.typename}",
                            "model",
                            args=standard_args,
                            modifier="",
                        ),
                        FunctionDeclaration(
                            f"static {reading_type.typename}::SensorJacobianT",
                            "jacobian",
                            args=standard_args,
                            modifier="",
                        ),
                        FunctionDeclaration(
                            f"static {reading_type.typename}::CovarianceT",
                            "covariance",
                            args=standard_args,
                            modifier="",
                        ),
                    ],
                )
            )

        body.append(
            ForwardClassDeclaration("class", "ExtendedKalmanFilterProcessModel")
        )

        ExtendedKalmanFilter = ClassDef(
            "class",
            "ExtendedKalmanFilter",
            bases=[],
            body=[
                Escape("public:"),
                UsingDeclaration(
                    "CovarianceT",
                    f"Eigen::Matrix<double, {generator.control_size}, {generator.control_size}>",
                ),
                UsingDeclaration(
                    "ProcessJacobianT",
                    f"Eigen::Matrix<double, {generator.state_size}, {generator.state_size}>",
                ),
                UsingDeclaration(
                    "ControlJacobianT",
                    f"Eigen::Matrix<double, {generator.state_size}, {generator.control_size}>",
                ),
                UsingDeclaration("ProcessModel", "ExtendedKalmanFilterProcessModel"),
                StateAndVariance_process_model(
                    enable_control=generator.enable_control(),
                    enable_calibration=generator.enable_calibration(),
                ),
                StateAndVariance_sensor_model(
                    enable_control=generator.enable_control(),
                    enable_calibration=generator.enable_calibration(),
                ),
                Templated(
                    [Arg("typename", "ReadingT")],
                    FunctionDef(
                        "std::optional<typename ReadingT::InnovationT>",
                        "innovations",
                        args=[],
                        modifier="",
                        body=[
                            FromFileTemplate(
                                "innovations.hpp",
                                inserts={},
                            )
                        ],
                    ),
                ),
                Escape("private:"),
                MemberDeclaration(
                    "std::unordered_map<SensorId, std::any>", "_innovations"
                ),
            ],
        )
        body.append(ExtendedKalmanFilter)

        def ExtendedKalmanFilterProcessModel_model(
            *, enable_calibration, enable_control
        ):
            args = [Arg("double", "dt"), Arg("const StateAndVariance&", "input")]
            if enable_calibration:
                args.append(Arg("const Calibration&", "input_calibration"))
            if enable_control:
                args.append(Arg("const Control&", "input_control"))
            return FunctionDeclaration("static State", "model", args=args, modifier="")

        def ExtendedKalmanFilterProcessModel_process_jacobian(
            *, enable_calibration, enable_control
        ):
            args = [Arg("double", "dt"), Arg("const StateAndVariance&", "input")]
            if enable_calibration:
                args.append(Arg("const Calibration&", "input_calibration"))
            if enable_control:
                args.append(Arg("const Control&", "input_control"))
            return FunctionDeclaration(
                "static typename ExtendedKalmanFilter::ProcessJacobianT",
                "process_jacobian",
                args=args,
                modifier="",
            )

        def ExtendedKalmanFilterProcessModel_control_jacobian(
            *, enable_calibration, enable_control
        ):
            args = [Arg("double", "dt"), Arg("const StateAndVariance&", "input")]
            if enable_calibration:
                args.append(Arg("const Calibration&", "input_calibration"))
            if enable_control:
                args.append(Arg("const Control&", "input_control"))
            return FunctionDeclaration(
                "static typename ExtendedKalmanFilter::ControlJacobianT",
                "control_jacobian",
                args=args,
                modifier="",
            )

        def ExtendedKalmanFilterProcessModel_covariance(
            *, enable_calibration, enable_control
        ):
            args = [Arg("double", "dt"), Arg("const StateAndVariance&", "input")]
            if enable_calibration:
                args.append(Arg("const Calibration&", "input_calibration"))
            if enable_control:
                args.append(Arg("const Control&", "input_control"))
            return FunctionDeclaration(
                "static typename ExtendedKalmanFilter::CovarianceT",
                "covariance",
                args=args,
                modifier="",
            )

        ExtendedKalmanFilterProcessModel = ClassDef(
            "class",
            "ExtendedKalmanFilterProcessModel",
            bases=[],
            body=[
                Escape("public:"),
                ExtendedKalmanFilterProcessModel_model(
                    enable_calibration=generator.enable_calibration(),
                    enable_control=generator.enable_control(),
                ),
                ExtendedKalmanFilterProcessModel_process_jacobian(
                    enable_calibration=generator.enable_calibration(),
                    enable_control=generator.enable_control(),
                ),
                ExtendedKalmanFilterProcessModel_control_jacobian(
                    enable_calibration=generator.enable_calibration(),
                    enable_control=generator.enable_control(),
                ),
                ExtendedKalmanFilterProcessModel_covariance(
                    enable_calibration=generator.enable_calibration(),
                    enable_control=generator.enable_control(),
                ),
            ],
        )
        body.append(ExtendedKalmanFilterProcessModel)
    else:  # enable_EKF == False
        Model = ClassDef(
            "class",
            "Model",
            bases=[],
            body=[
                Escape("public:"),
                State_model(
                    enable_control=generator.enable_control(),
                    enable_calibration=generator.enable_calibration(),
                ),
            ],
        )

        body.append(Model)

    namespace = Namespace(name=generator.namespace, body=body)
    includes = [
        "#include <Eigen/Dense>    // Matrix",
    ]
    if generator.enable_EKF:
        includes.append("#include <any>")
        includes.append("#include <optional>")
    header = HeaderFile(pragma=True, includes=includes, namespaces=[namespace])
    return header.compile(CompileState(indent=2))


def source_from_ast(*, generator):
    body = [
        ConstructorDefinition(
            "State", args=[], initializer_list=[("data", "DataT::Zero()")]
        ),
        ConstructorDefinition(
            "State",
            args=[Arg("const StateOptions&", "options")],
            initializer_list=[
                (
                    "data",
                    ", ".join(f"options.{name}" for name in generator.arglist_state),
                ),
            ],
        ),
    ]

    if generator.enable_calibration():
        CalibrationDefaultConstructor = ConstructorDefinition(
            "Calibration", args=[], initializer_list=[("data", "DataT::Zero()")]
        )
        CalibrationConstructor = ConstructorDefinition(
            "Calibration",
            args=[Arg("const CalibrationOptions&", "options")],
            initializer_list=[
                (
                    "data",
                    ", ".join(
                        f"options.{name}" for name in generator.arglist_calibration
                    ),
                ),
            ],
        )
        body.append(CalibrationDefaultConstructor)
        body.append(CalibrationConstructor)

    if generator.enable_control():
        ControlDefaultConstructor = ConstructorDefinition(
            "Control", args=[], initializer_list=[("data", "DataT::Zero()")]
        )
        ControlConstructor = ConstructorDefinition(
            "Control",
            args=[Arg("const ControlOptions&", "options")],
            initializer_list=[
                (
                    "data",
                    ", ".join(f"options.{name}" for name in generator.arglist_control),
                ),
            ],
        )
        body.append(ControlDefaultConstructor)
        body.append(ControlConstructor)

    if generator.enable_EKF:
        standard_args = [Arg("double", "dt"), Arg("const StateAndVariance&", "input")]
        if generator.enable_calibration():
            standard_args.append(Arg("const Calibration&", "input_calibration"))
        if generator.enable_control():
            standard_args.append(Arg("const Control&", "input_control"))
        EKF_process_model = FunctionDef(
            "StateAndVariance",
            "ExtendedKalmanFilter::process_model",
            modifier="",
            args=standard_args,
            body=[
                FromFileTemplate(
                    "process_model.cpp",
                    inserts={
                        "enable_control": generator.enable_control(),
                        "enable_calibration": generator.enable_calibration(),
                    },
                ),
            ],
        )
        body.append(EKF_process_model)
        EKFPM_model = FunctionDef(
            "State",
            "ExtendedKalmanFilterProcessModel::model",
            modifier="",
            args=standard_args,
            body=[
                Escape(generator.process_model_body()),
            ],
        )
        body.append(EKFPM_model)
        EKFPM_process_jacobian = FunctionDef(
            "typename ExtendedKalmanFilter::ProcessJacobianT",
            "ExtendedKalmanFilterProcessModel::process_jacobian",
            modifier="",
            args=standard_args,
            body=[
                Escape(generator.process_jacobian_body()),
            ],
        )
        body.append(EKFPM_process_jacobian)
        EKFPM_control_jacobian = FunctionDef(
            "typename ExtendedKalmanFilter::ControlJacobianT",
            "ExtendedKalmanFilterProcessModel::control_jacobian",
            modifier="",
            args=standard_args,
            body=[
                Escape(generator.control_jacobian_body()),
            ],
        )
        body.append(EKFPM_control_jacobian)
        EKFPM_covariance = FunctionDef(
            "typename ExtendedKalmanFilter::CovarianceT",
            "ExtendedKalmanFilterProcessModel::covariance",
            modifier="",
            args=standard_args,
            body=[
                Escape(generator.control_covariance_body()),
            ],
        )
        body.append(EKFPM_covariance)

        for reading_type in generator.reading_types():
            standard_args = [
                Arg("const StateAndVariance&", "input"),
            ]
            if generator.enable_calibration():
                standard_args.append(Arg("const Calibration&", "input_calibration"))
            standard_args.append(
                Arg(f"const {reading_type.typename}&", "input_reading")
            )
            ReadingDefaultConstructor = ConstructorDefinition(
                reading_type.typename,
                args=[],
                initializer_list=[("data", "DataT::Zero()")],
            )
            body.append(ReadingDefaultConstructor)
            ReadingConstructor = ConstructorDefinition(
                reading_type.typename,
                args=[Arg(f"const {reading_type.typename}Options&", "options")],
                initializer_list=[
                    (
                        "data",
                        ", ".join(
                            f"options.{name}"
                            for name in sorted(
                                list(reading_type.sensor_model_mapping.keys())
                            )
                        ),
                    )
                ],
            )
            body.append(ReadingConstructor)
            ReadingSensorModel_model = FunctionDef(
                reading_type.typename,
                f"{reading_type.typename}SensorModel::model",
                modifier="",
                args=standard_args,
                body=[Escape(reading_type.SensorModel_model_body)],
            )
            body.append(ReadingSensorModel_model)
            ReadingSensorModel_covariance = FunctionDef(
                f"{reading_type.typename}::CovarianceT",
                f"{reading_type.typename}SensorModel::covariance",
                modifier="",
                args=standard_args,
                body=[Escape(reading_type.SensorModel_covariance_body)],
            )
            body.append(ReadingSensorModel_covariance)
            ReadingSensorModel_jacobian = FunctionDef(
                f"{reading_type.typename}::SensorJacobianT",
                f"{reading_type.typename}SensorModel::jacobian",
                modifier="",
                args=standard_args,
                body=[Escape(reading_type.SensorModel_jacobian_body)],
            )
            body.append(ReadingSensorModel_jacobian)
    else:  # generator.enable_EKF == False
        standard_args = [Arg("double", "dt"), Arg("const State&", "input_state")]
        if generator.enable_calibration():
            standard_args.append(Arg("const Calibration&", "input_calibration"))
        if generator.enable_control():
            standard_args.append(Arg("const Control&", "input_control"))
        Model_model = FunctionDef(
            "State",
            "Model::model",
            modifier="",
            args=standard_args,
            body=[
                Escape(generator.model_body()),
            ],
        )
        body.append(Model_model)

    namespace = Namespace(name=generator.namespace, body=body)
    includes = [f"#include <{generator.header_include}>"]
    src = SourceFile(includes=includes, namespaces=[namespace])
    return src.compile(CompileState(indent=2))


def _compile_impl(args, *, generator):
    # Compilation

    if args.header is None or args.source is None:
        print('"Rendering" to stdout')
        return CppCompileResult(
            success=False,
        )

    header_str = "\n".join(header_from_ast(generator=generator))
    source_str = "\n".join(source_from_ast(generator=generator))

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


def compile(symbolic_model, calibration_map=None, *, config=None):
    if config is None:
        config = Config()
    elif isinstance(config, dict):
        config = Config(**config)

    if calibration_map is None:
        calibration_map = {}

    args = _compile_argparse()

    if args.header is None:
        logger.warning("No Header specified, so output to stdout")
    generator = _generate_model_function_bodies(
        header_location=args.header,
        namespace=args.namespace,
        symbolic_model=symbolic_model,
        calibration_map=calibration_map,
        config=config,
    )

    return _compile_impl(args, generator=generator)


def compile_ekf(
    state_model,
    process_noise,
    sensor_models,
    sensor_noises,
    calibration_map=None,
    *,
    config=None,
):
    if config is None:
        config = Config()
    elif isinstance(config, dict):
        config = Config(**config)

    if calibration_map is None:
        calibration_map = {}

    common.model_validation(
        state_model,
        process_noise,
        sensor_models,
        extra_validation=config.extra_validation,
    )

    args = _compile_argparse()

    if args.header is None:
        logger.warning("No Header specified, so output to stdout")
    generator = _generate_ekf_function_bodies(
        header_location=args.header,
        namespace=args.namespace,
        state_model=state_model,
        process_noise=process_noise,
        sensor_models=sensor_models,
        sensor_noises=sensor_noises,
        calibration_map=calibration_map,
        config=config,
    )

    return _compile_impl(args, generator=generator)
