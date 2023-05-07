import ast
from dataclasses import dataclass
from itertools import chain
from typing import Any, List, Optional


@dataclass
class CompileState:
    indent: int = 0


class BaseAst(ast.AST):
    def __init__(self):
        self.lineno = None
        self.col_offset = None
        self.end_lineno = None
        self.end_col_offset = None

    def compile(self, options: CompileState, **kwargs):
        raise NotImplementedError()

    def indent(self, options: CompileState):
        return " " * options.indent


def autoindent(compile_func):
    def wrapped(self, options: CompileState, **kwargs):
        for line in compile_func(self, options, **kwargs):
            yield " " * options.indent + line

    # TODO(buck): wrapper helper function
    wrapped.__name__ = compile_func.__name__
    return wrapped


@dataclass
class Namespace(BaseAst):
    _fields = ("name", "body")

    name: str
    body: List[Any]

    def compile(self, options: CompileState, **kwargs):
        yield f"namespace {self.name} {{"

        for component in self.body:
            yield from component.compile(options, **kwargs)

        yield f"}} // namespace {self.name}"


@dataclass
class HeaderFile(BaseAst):
    _fields = ("pragma", "includes", "namespaces")

    # pragma: true or false. If true, include #pragma once
    pragma: bool
    includes: List[str]
    namespaces: List[Namespace]

    def compile(self, options: CompileState, **kwargs):
        if self.pragma:
            yield "#pragma once"
            yield ""

        for include in self.includes:
            yield include
        yield ""

        for namespace in self.namespaces:
            yield from namespace.compile(options, **kwargs)


@dataclass
class ClassDef(BaseAst):
    _fields = ("tag", "name", "bases", "body")

    # tag: one of "struct", "class"
    tag: str
    name: str
    bases: List[str]
    body: List[Any]

    @autoindent
    def compile(self, options: CompileState, **kwargs):
        bases_str = ""
        if len(self.bases) > 0:
            raise NotImplementedError()

        yield f"{self.tag} {self.name} {bases_str} {{"

        for component in self.body:
            yield from component.compile(options, classname=self.name, **kwargs)

        yield "}\n"


@dataclass
class MemberDeclaration(BaseAst):
    _fields = ("type_", "name", "value")

    type_: str
    name: str
    value: Optional[Any] = None

    @autoindent
    def compile(self, options: CompileState, **kwargs):
        value_str = ""
        if self.value is not None:
            value_str = f"= {self.value}"
        yield f"{self.type_} {self.name} {value_str};"


@dataclass
class UsingDeclaration(BaseAst):
    _fields = ("name", "type_")

    name: str
    type_: str

    @autoindent
    def compile(self, options: CompileState, **kwargs):
        yield f"using {self.name} = {self.type_};"


class ConstructorDeclaration(BaseAst):
    _fields = ("args",)

    def __init__(self, *args):
        self.args = args

    @autoindent
    def compile(self, options: CompileState, classname: str, **kwargs):
        if len(self.args) > 0:
            yield f"{classname}("
            for arg in self.args:
                for line in arg.compile(options, classname=classname, **kwargs):
                    yield line + ","
            yield ");"
        else:
            yield f"{classname}();"


@dataclass
class Arg(BaseAst):
    _fields = ("type_", "name")

    type_: str
    name: str

    @autoindent
    def compile(self, options: CompileState, **kwargs):
        yield f"{self.type_} {self.name}"


@dataclass
class FunctionDef(BaseAst):
    _fields = ("return_type", "name", "args", "modifier", "body")

    return_type: str
    name: str
    args: List[Any]
    modifier: str
    body: List[Any]

    @autoindent
    def compile(self, options: CompileState, **kwargs):
        if len(self.args) > 0:
            yield f"{self.return_type} {self.name}("
            for arg in self.args:
                yield from arg.compile(options, **kwargs)
            yield f") {self.modifier} {{"
        else:
            yield f"{self.return_type} {self.name}() {self.modifier} {{"

        for component in self.body:
            yield from component.compile(options, **kwargs)

        yield "}"


@dataclass
class Return(BaseAst):
    _fields = ("value",)

    value: str

    @autoindent
    def compile(self, options: CompileState, **kwargs):
        yield f"return {self.value};"


StateOptions = ClassDef(
    "struct",
    "StateOptions",
    bases=[],
    body=[
        MemberDeclaration("double", "CON_ori_pitch", 0.0),
        MemberDeclaration("double", "CON_ori_roll", 0.0),
        MemberDeclaration("double", "CON_ori_yaw", 0.0),
        MemberDeclaration("double", "CON_pos_pos_x", 0.0),
        MemberDeclaration("double", "CON_pos_pos_y", 0.0),
        MemberDeclaration("double", "CON_pos_pos_z", 0.0),
        MemberDeclaration("double", "CON_vel_x", 0.0),
        MemberDeclaration("double", "CON_vel_y", 0.0),
        MemberDeclaration("double", "CON_vel_z", 0.0),
    ],
)

State = ClassDef(
    "struct",
    "State",
    bases=[],
    body=[
        MemberDeclaration("static constexpr size_t", "rows", 9),
        MemberDeclaration("static constexpr size_t", "cols", 1),
        # TODO(buck): Eigen::Matrix<...> can be split into its own structure
        UsingDeclaration("DataT", "Eigen::Matrix<double, rows, cols>"),
        ConstructorDeclaration(),  # No args constructor gets default constructor
        ConstructorDeclaration(Arg("const StateOptions&", "options")),
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
                for idx, name in enumerate(
                    [
                        "CON_ori_pitch",
                        "CON_ori_roll",
                        "CON_ori_yaw",
                        "CON_pos_pos_x",
                        "CON_pos_pos_y",
                        "CON_pos_pos_z",
                        "CON_vel_x",
                        "CON_vel_y",
                        "CON_vel_z",
                    ]
                )
            ]
        )
    )
    + [
        MemberDeclaration("DataT", "data", "DataT::Zero()"),
    ],
)

Covariance = ClassDef(
    "struct",
    "Covariance",
    bases=[],
    body=[
        UsingDeclaration("DataT", "Eigen::Matrix<double, 9, 9>"),
        ConstructorDeclaration(),  # No args constructor gets default constructor
        ConstructorDeclaration(Arg("const StateOptions&", "options")),
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
                for idx, name in enumerate(
                    [
                        "CON_ori_pitch",
                        "CON_ori_roll",
                        "CON_ori_yaw",
                        "CON_pos_pos_x",
                        "CON_pos_pos_y",
                        "CON_pos_pos_z",
                        "CON_vel_x",
                        "CON_vel_y",
                        "CON_vel_z",
                    ]
                )
            ]
        )
    )
    + [
        MemberDeclaration("DataT", "data", "DataT::Identity()"),
    ],
)

ControlOptions = ClassDef(
    "struct",
    "ControlOptions",
    bases=[],
    body=[
        MemberDeclaration("double", "IMU_reading_acc_x", 0.0),
        MemberDeclaration("double", "IMU_reading_acc_y", 0.0),
        MemberDeclaration("double", "IMU_reading_acc_z", 0.0),
        MemberDeclaration("double", "IMU_reading_pitch_rate", 0.0),
        MemberDeclaration("double", "IMU_reading_roll_rate", 0.0),
        MemberDeclaration("double", "IMU_reading_yaw_rate", 0.0),
    ],
)

Control = ClassDef(
    "struct",
    "Control",
    bases=[],
    body=[
        MemberDeclaration("static constexpr size_t", "rows", 6),
        MemberDeclaration("static constexpr size_t", "cols", 1),
        UsingDeclaration("DataT", "Eigen::Matrix<double, rows, cols>"),
        ConstructorDeclaration(),  # No args constructor gets default constructor
        ConstructorDeclaration(Arg("const ControlOptions&", "options")),
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
                for idx, name in enumerate(
                    [
                        "IMU_reading_acc_x",
                        "IMU_reading_acc_y",
                        "IMU_reading_acc_z",
                        "IMU_reading_pitch_rate",
                        "IMU_reading_roll_rate",
                        "IMU_reading_yaw_rate",
                    ]
                )
            ]
        )
    )
    + [
        MemberDeclaration("DataT", "data", "DataT::Zero()"),
    ],
)

namespace = Namespace(
    name="featuretest", body=[StateOptions, State, Covariance, ControlOptions, Control]
)

includes = [
    "#include <Eigen/Dense>    // Matrix",
    "#include <any>            // any",
    "#include <cstddef>        // size_t",
    "#include <iostream>       // std::cout, debugging",
    "#include <optional>       // optional",
    "#include <unordered_map>  // unordered_map",
]
header = HeaderFile(pragma=True, includes=includes, namespaces=[namespace])

print("\n".join(header.compile(CompileState(indent=2))))
