import ast
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from jinja2 import Environment, FileSystemLoader, select_autoescape
from jinja2.exceptions import TemplateNotFound


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
class Arg(BaseAst):
    _fields = ("type_", "name")

    type_: str
    name: str

    @autoindent
    def compile(self, options: CompileState, **kwargs):
        yield f"{self.type_} {self.name}"


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
class SourceFile(BaseAst):
    _fields = ("includes", "namespaces")

    includes: List[str]
    namespaces: List[Namespace]

    def compile(self, options: CompileState, **kwargs):
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

        yield "};\n"


@dataclass
class EnumClassDef(BaseAst):
    _fields = ("name", "members")

    name: str
    members: List[str]

    @autoindent
    def compile(self, options: CompileState, **kwargs):
        yield f"enum class {self.name} {{"
        for member in self.members:
            yield f"{' ' * options.indent}{member},"
        yield "};\n"


@dataclass
class ForwardClassDeclaration(BaseAst):
    _fields = ("tag", "name")

    # tag: one of "struct", "class"
    tag: str
    name: str

    @autoindent
    def compile(self, options: CompileState, **kwargs):
        yield f"{self.tag} {self.name};"


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
            value_str = f" = {self.value}"
        yield f"{self.type_} {self.name}{value_str};"


@dataclass
class UsingDeclaration(BaseAst):
    _fields = ("name", "type_")

    name: str
    type_: str

    @autoindent
    def compile(self, options: CompileState, **kwargs):
        yield f"using {self.name} = {self.type_};"


@dataclass
class ConstructorDeclaration(BaseAst):
    _fields = ("args",)

    args: Optional[List[Arg]] = None

    @autoindent
    def compile(self, options: CompileState, classname: str, **kwargs):
        if self.args is None:
            self.args = []

        if len(self.args) == 0:
            yield f"{classname}();"
        elif len(self.args) == 1:
            # specialization for "short" functions
            argstr = "".join(
                self.args[0].compile(options, classname=classname, **kwargs)
            ).strip()
            yield f"{classname}({argstr});"
        else:
            yield f"{classname}("
            for arg in self.args[:-1]:
                for line in arg.compile(options, classname=classname, **kwargs):
                    yield line + ","
            yield from self.args[-1].compile(options, classname=classname, **kwargs)
            yield ");"


@dataclass
class ConstructorDefinition(BaseAst):
    _fields = (
        "classname",
        "args",
        "initializer_list",
    )

    classname: str
    args: Optional[List[Arg]] = None
    initializer_list: Optional[List[Tuple[str, str]]] = None

    @autoindent
    def compile(self, options: CompileState, **kwargs):
        if self.args is None:
            self.args = []
        if self.initializer_list is None:
            self.initializer_list = []

        initializer_list_str = ", ".join(f"{k}({v})" for k, v in self.initializer_list)
        if initializer_list_str != "":
            initializer_list_str = f" : {initializer_list_str}"

        if len(self.args) == 0:
            yield f"{self.classname}::{self.classname}(){initializer_list_str} {{}}"
        elif len(self.args) == 1:
            # specialization for "short" functions
            argstr = "".join(self.args[0].compile(options, **kwargs)).strip()
            yield f"{self.classname}::{self.classname}({argstr}){initializer_list_str} {{}}"
        else:
            yield f"{self.classname}::{self.classname}("
            for arg in self.args[:-1]:
                for line in arg.compile(options, **kwargs):
                    yield line + ","
            yield from self.args[-1].compile(options, **kwargs)
            yield "){initializer_list_str} {}"


@dataclass
class FunctionDef(BaseAst):
    _fields = ("return_type", "name", "args", "modifier", "body")

    return_type: str
    name: str
    args: List[Arg]
    modifier: str
    body: List[Any]

    @autoindent
    def compile(self, options: CompileState, **kwargs):
        modifier_str = ""
        if len(self.modifier) > 0:
            modifier_str = " " + self.modifier

        if len(self.args) == 0:
            yield f"{self.return_type} {self.name}(){modifier_str} {{"
        elif len(self.args) == 1:
            # specialization for "short" functions
            argstr = "".join(self.args[0].compile(options, **kwargs)).strip()
            yield f"{self.return_type} {self.name}({argstr}){modifier_str} {{"
        else:
            yield f"{self.return_type} {self.name}("
            for arg in self.args[:-1]:
                for line in arg.compile(options, **kwargs):
                    yield line + ","
            yield from self.args[-1].compile(options, **kwargs)
            yield f"){modifier_str} {{"

        for component in self.body:
            yield from component.compile(options, **kwargs)

        yield "}"


@dataclass
class FunctionDeclaration(BaseAst):
    _fields = ("return_type", "name", "args", "modifier")

    return_type: str
    name: str
    args: List[Arg]
    modifier: str

    @autoindent
    def compile(self, options: CompileState, **kwargs):
        modifier_str = ""
        if len(self.modifier) > 0:
            modifier_str = " " + self.modifier

        if len(self.args) == 0:
            yield f"{self.return_type} {self.name}(){modifier_str};"
        elif len(self.args) == 1:
            # specialization for "short" functions
            argstr = "".join(self.args[0].compile(options, **kwargs)).strip()
            yield f"{self.return_type} {self.name}({argstr}){modifier_str};"
        else:
            yield f"{self.return_type} {self.name}("
            for arg in self.args[:-1]:
                for line in arg.compile(options, **kwargs):
                    yield line + ","
            yield from self.args[-1].compile(options, **kwargs)
            yield f"){modifier_str};"


@dataclass
class Return(BaseAst):
    _fields = ("value",)

    value: str

    @autoindent
    def compile(self, options: CompileState, **kwargs):
        yield f"return {self.value};"


@dataclass
class If(BaseAst):
    _fields = ("test", "body", "orelse")

    test: str
    body: List[Any]
    orelse: List[Any]

    @autoindent
    def compile(self, options: CompileState, **kwargs):
        yield f"if ({self.test}) {{"
        for component in self.body:
            yield from component.compile(options, **kwargs)

        if len(self.orelse) == 0:
            yield "}"
        else:
            yield "} else {"
            for component in self.orelse:
                yield from component.component(options, **kwargs)
            yield "}"


@dataclass
class Templated(BaseAst):
    _fields = ("template_args", "templated")

    template_args: List[Arg]
    templated: Any

    @autoindent
    def compile(self, options: CompileState, **kwargs):
        if len(self.template_args) == 0:
            yield "template <>"
        elif len(self.template_args) == 1:
            argstr = "".join(self.template_args[0].compile(options, **kwargs)).strip()
            yield f"template <{argstr}>"
        else:
            yield "template <"
            for arg in self.template_args:
                yield from arg.compile(options, **kwargs)
            yield ">"
        yield from self.templated.compile(options, **kwargs)


@dataclass
class FromFileTemplate(BaseAst):
    _fields = ("template_options", "name", "inserts")

    template_options: Union[str, os.PathLike]
    name: str
    inserts: Optional[Dict[str, Any]] = None

    @autoindent
    def compile(self, options: CompileState, **kwargs):
        if self.inserts is None:
            self.inserts = {}

        templates_base_path = self.template_options.base
        # jinja
        env = Environment(
            loader=FileSystemLoader(templates_base_path), autoescape=select_autoescape()
        )

        # load template
        try:
            template = env.get_template(self.name)
        except TemplateNotFound:
            print("Debugging TemplateNotFound")
            print("Trying to scandir")
            with os.scandir(templates_base_path) as it:
                if len(list(it)) == 0:
                    print("No Paths in scandir")
                    raise

            print("Walking")
            for root, _, files in os.walk(templates_base_path):
                depth = len(root.split("/"))
                print("{}Root: {!s}".format(" " * depth, root))
                for filename in files:
                    print("{}  - {!s}".format(" " * depth, filename))
            print("End Walk")
            raise

        # substitute
        template_str = template.render(**self.inserts)
        # for line in substituted_template, yield line
        yield from template_str.split("\n")


@dataclass
class Escape(BaseAst):
    _fields = ("string",)

    string: str

    def compile(self, options: CompileState, **kwargs):
        yield self.string
