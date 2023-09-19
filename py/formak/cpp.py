import argparse
import logging
from collections import namedtuple
from dataclasses import dataclass
from itertools import count
from typing import Any, Iterable, List, Optional, Tuple

from formak.ast_tools import (
    BaseAst,
    ClassDef,
    CompileState,
    ForwardClassDeclaration,
    HeaderFile,
    MemberDeclaration,
    Namespace,
    Return,
    SourceFile,
)
from formak.exceptions import ModelConstructionError
from sympy import Symbol, ccode, cse, diff, simplify

from formak import ast_fragments as fragments
from formak import common

DEFAULT_MODULES = ("scipy", "numpy", "math")

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """
    Options for generating C++.

    common_subexpression_elimination:
        Remove common shared computation
    extra_validation:
        Catch errors earlier in exchange for increased compute time
    """

    common_subexpression_elimination: bool = True
    extra_validation: bool = False
    max_dt_sec: float = 0.1
    innovation_filtering: float = 5.0

    def ccode(self):
        if self.max_dt_sec < 1e-9:
            raise ValueError(
                "Please specify Config(max_dt_sec=...) >= 1e-9. Currently {self.max_dt_sec}"
            )

        return Namespace(
            name="cpp",
            body=[
                ClassDef(
                    "struct",
                    "Config",
                    bases=[],
                    body=[
                        MemberDeclaration(
                            "static constexpr bool",
                            "common_subexpression_elimination",
                            "true"
                            if self.common_subexpression_elimination
                            else "false",
                        ),
                        MemberDeclaration(
                            "static constexpr bool",
                            "extra_validation",
                            "true" if self.extra_validation else "false",
                        ),
                        MemberDeclaration(
                            "static constexpr double", "max_dt_sec", self.max_dt_sec
                        ),
                        MemberDeclaration(
                            "static constexpr double",
                            "innovation_filtering",
                            self.innovation_filtering
                            if self.innovation_filtering
                            else 0.0,
                        ),
                    ],
                )
            ],
        )


@dataclass
class CppCompileResult:
    success: bool
    header_path: Optional[str] = None
    source_path: Optional[str] = None


class BasicBlock:
    """
    A run of statements without control flow.

    All statements can be reordered or changed to improve performance.
    """

    def __init__(
        self, *, statements: List[Tuple[str, Any]], indent: int, config: Config
    ):
        # should be Tuple[str, sympy expression]
        statements = list(statements)
        self._targets = [k for k, _ in statements]
        self._exprs = [v for _, v in statements]
        self._indent = indent
        self._config = config

    def __len__(self):
        return len(self._exprs)

    def compile(self):
        prefix = []
        body = self._exprs

        if self._config.common_subexpression_elimination:
            prefix, body = cse(body, symbols=(Symbol(f"_t{i}") for i in count()))

        # Note: The list of statements is ordered and can get CSE or reordered
        # within the block because we know it is straight calculation without
        # control flow (a basic block)
        for target, expr in prefix:
            assert isinstance(target, Symbol)
            if self._config.common_subexpression_elimination:
                expr = simplify(expr)
            cc_expr = ccode(expr)
            yield MemberDeclaration("double", target, cc_expr)

        for target, expr in zip(self._targets, body):
            if self._config.common_subexpression_elimination:
                expr = simplify(expr)
            cc_expr = ccode(expr)
            yield MemberDeclaration("", target, cc_expr)


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
        self.config = config
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

        self.State = common.named_vector("State", self.arglist_state)
        self.Control = common.named_vector("Control", self.arglist_control)
        self.Calibration = common.named_vector("Calibration", self.arglist_calibration)

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
            statements=self._translate_model(symbolic_model), indent=4, config=config
        )

        self._return = self._translate_return()

    def _translate_model(self, symbolic_model):
        subs_set = (
            [
                (
                    member,
                    Symbol("state.{}()".format(member)),
                )
                for member in self.arglist_state
            ]
            + [
                (
                    member,
                    Symbol("calibration.{}()".format(member)),
                )
                for member in self.arglist_calibration
            ]
            + [
                (
                    member,
                    Symbol("control.{}()".format(member)),
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
        yield from self._model.compile()
        yield Return(self._return)

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

        self.State = common.named_vector("State", self.arglist_state)
        self.Covariance = common.named_covariance("Covariance", self.arglist_state)
        self.Control = common.named_vector("Control", self.arglist_control)
        self.Calibration = common.named_vector("Calibration", self.arglist_calibration)

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
            statements=self._translate_process_model(state_model),
            indent=4,
            config=config,
        )
        self._process_jacobian = BasicBlock(
            statements=self._translate_process_jacobian(state_model),
            indent=4,
            config=config,
        )

        self._control_jacobian = BasicBlock(
            statements=self._translate_control_jacobian(state_model),
            indent=4,
            config=config,
        )
        self._control_covariance = BasicBlock(
            statements=self._translate_control_covariance(process_noise),
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
                    Symbol("state.state.{}()".format(member)),
                )
                for member in self.arglist_state
            ]
            + [
                (
                    member,
                    Symbol("calibration.{}()".format(member)),
                )
                for member in self.arglist_calibration
            ]
            + [
                (
                    member,
                    Symbol("control.{}()".format(member)),
                )
                for member in self.arglist_control
            ]
        )

        for a in self.arglist_state:
            expr_before = symbolic_model.state_model[a]
            expr_after = expr_before.subs(subs_set)
            yield f"double {a.name}", expr_after

    def process_model_body(self):
        yield from self._process_model.compile()
        yield Return(self._return)

    def _translate_process_jacobian(self, symbolic_model):
        subs_set = (
            [
                (
                    member,
                    Symbol("state.state.{}()".format(member)),
                )
                for member in self.arglist_state
            ]
            + [
                (
                    member,
                    Symbol("calibration.{}()".format(member)),
                )
                for member in self.arglist_calibration
            ]
            + [
                (
                    member,
                    Symbol("control.{}()".format(member)),
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
        yield MemberDeclaration("ExtendedKalmanFilter::ProcessJacobianT", "jacobian")
        yield from self._process_jacobian.compile()
        yield Return("jacobian")

    def _translate_control_jacobian(self, symbolic_model):
        subs_set = (
            [
                (
                    member,
                    Symbol("state.state.{}()".format(member)),
                )
                for member in self.arglist_state
            ]
            + [
                (
                    member,
                    Symbol("calibration.{}()".format(member)),
                )
                for member in self.arglist_calibration
            ]
            + [
                (
                    member,
                    Symbol("control.{}()".format(member)),
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
        yield MemberDeclaration("ExtendedKalmanFilter::ControlJacobianT", "jacobian")
        yield from self._control_jacobian.compile()
        yield Return("jacobian")

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
        yield MemberDeclaration("ExtendedKalmanFilter::CovarianceT", "covariance")
        yield from self._control_covariance.compile()
        yield Return("covariance")

    def enable_calibration(self):
        return self.calibration_size > 0

    def _translate_sensor_model(self, sensor_model_mapping):
        subs_set = [
            (
                member,
                Symbol("state.state.{}()".format(member)),
            )
            for member in self.arglist_state
        ] + [
            (
                member,
                Symbol("calibration.{}()".format(member)),
            )
            for member in self.arglist_calibration
        ]
        for predicted_reading, model in sorted(list(sensor_model_mapping.items())):
            expr_before = model
            expr_after = expr_before.subs(subs_set)
            yield f"double {predicted_reading}", expr_after

    def reading_types(self, verbose=False):
        for name, sensor_model_mapping, sensor_noise in self.sensorlist:
            typename = name.title()
            identifier = f"SensorId::{name.upper()}"
            arglist_sensor = sorted(list(sensor_model_mapping.keys()))
            members = "\n".join(
                "double& %s() { return data(%d, 0); }" % (name, idx)
                for idx, name in enumerate(arglist_sensor)
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
                statements=self._translate_sensor_model(sensor_model_mapping),
                indent=4,
                config=self.config,
            )
            return_ = Return(
                "{}Options{{".format(typename)
                + ", ".join(
                    str(reading)
                    for reading in sorted(list(sensor_model_mapping.keys()))
                )
                + "}"
            )
            SensorModel_model_body = list(body.compile()) + [return_]

            SensorCovariance = common.named_covariance(
                f"{name}Covariance", arglist_sensor
            )

            SensorModel_covariance_body = self._translate_sensor_covariance(
                typename, SensorCovariance.from_dict(sensor_noise)
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
                Symbol("state.state.{}()".format(member)),
            )
            for member in self.arglist_state
        ] + [
            (
                member,
                Symbol("calibration.{}()".format(member)),
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
        yield MemberDeclaration(f"{typename}::SensorJacobianT", "jacobian")
        yield from BasicBlock(
            statements=self._translate_sensor_jacobian_impl(sensor_model_mapping),
            indent=4,
            config=self.config,
        ).compile()
        yield Return("jacobian")

    def _translate_sensor_covariance_impl(self, covariance):
        rows, cols = covariance.shape
        for i in range(rows):
            for j in range(cols):
                yield f"covariance({i}, {j})", covariance.data[i, j]

    def _translate_sensor_covariance(self, typename, covariance):
        yield MemberDeclaration(f"{typename}::CovarianceT", "covariance")
        yield from BasicBlock(
            statements=self._translate_sensor_covariance_impl(covariance),
            indent=4,
            config=self.config,
        ).compile()
        yield Return("covariance")


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


def _header_body(*, generator) -> Iterable[BaseAst]:
    yield generator.config.ccode()
    yield fragments.StateOptions(generator)
    yield fragments.State(generator)

    if generator.enable_control():
        yield fragments.ControlOptions(generator)
        yield fragments.Control(generator)

    if generator.enable_calibration():
        yield fragments.CalibrationOptions(generator)
        yield fragments.Calibration(generator)

    if generator.enable_EKF:
        yield fragments.Covariance(generator)
        yield fragments.StateAndVariance(generator)
        yield fragments.SensorId(generator)
        yield ForwardClassDeclaration("class", "ExtendedKalmanFilterProcessModel")
        yield ForwardClassDeclaration("struct", "StampedReadingBase")

        yield fragments.ExtendedKalmanFilter(generator)

        yield fragments.ExtendedKalmanFilterProcessModel(generator)
        yield fragments.StampedReadingBase(generator)

        for reading_type in generator.reading_types():
            yield ForwardClassDeclaration(
                "struct", f"{reading_type.typename}SensorModel"
            )
            yield fragments.ReadingOptions(reading_type)
            yield fragments.Reading(generator, reading_type)
            yield fragments.ReadingSensorModel(generator, reading_type)

    else:  # enable_EKF == False
        yield fragments.Model(generator)


def header_from_ast(*, generator) -> str:
    namespace = Namespace(
        name=generator.namespace, body=_header_body(generator=generator)
    )
    includes = [
        "#include <Eigen/Dense>    // Matrix",
        "#include <formak/innovation_filtering.h>",
    ]
    if generator.enable_EKF:
        includes.append("#include <any>")
        includes.append("#include <optional>")
        includes.append("#include <type_traits>")  # false_type
    header = HeaderFile(pragma=True, includes=includes, namespaces=[namespace])
    return header.compile(CompileState(indent=2))


def _source_body(*, generator):
    yield fragments.StateDefaultConstructor()
    yield fragments.StateOptionsConstructor(generator)

    if generator.enable_calibration():
        yield fragments.CalibrationDefaultConstructor()
        yield fragments.CalibrationConstructor(generator)

    if generator.enable_control():
        yield fragments.ControlDefaultConstructor()
        yield fragments.ControlConstructor(generator)

    if generator.enable_EKF:
        yield fragments.EKF_process_model(generator)
        yield fragments.EKFPM_model(generator)
        yield fragments.EKFPM_process_jacobian(generator)
        yield fragments.EKFPM_control_jacobian(generator)
        yield fragments.EKFPM_covariance(generator)

        for reading_type in generator.reading_types():
            yield fragments.ReadingDefaultConstructor(reading_type)
            yield fragments.ReadingConstructor(reading_type)
            yield fragments.ReadingSensorModel_model(generator, reading_type)
            yield fragments.ReadingSensorModel_covariance(generator, reading_type)
            yield fragments.ReadingSensorModel_jacobian(generator, reading_type)
    else:  # generator.enable_EKF == False
        yield fragments.Model_model(generator)


def source_from_ast(*, generator):
    namespace = Namespace(
        name=generator.namespace, body=_source_body(generator=generator)
    )
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

    with open(args.header, "w") as header_file, open(args.source, "w") as source_file:
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
        calibration_map=calibration_map,
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
