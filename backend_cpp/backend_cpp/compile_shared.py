import argparse
from typing import Iterable

from backend_cpp import ast_fragments as fragments
from backend_cpp.ast import (
    BaseAst,
    CompileState,
    ForwardClassDeclaration,
    HeaderFile,
    Namespace,
    SourceFile,
)
from backend_cpp.compile_result import CppCompileResult


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
