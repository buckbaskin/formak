import logging

from itertools import chain

from formak.ast_tools import (
    Arg,
    ClassDef,
    CompileState,
    ConstructorDeclaration,
    ConstructorDefinition,
    Escape,
    FunctionDef,
    FunctionDeclaration,
    HeaderFile,
    MemberDeclaration,
    Namespace,
    Return,
    SourceFile,
    UsingDeclaration,
)


MEMBERS = [
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


def header_definition():
    StateOptions = ClassDef(
        "struct",
        "StateOptions",
        bases=[],
        body=[MemberDeclaration("double", member, 0.0) for member in MEMBERS],
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
                    for idx, name in enumerate(MEMBERS)
                ]
            )
        )
        + [
            MemberDeclaration("DataT", "data", "DataT::Zero()"),
        ],
    )
    func = FunctionDeclaration(
        "State",
        "elementwise_clamp",
        args=[
            Arg("const State&", "lower"),
            Arg("const State&", "value"),
            Arg("const State&", "upper"),
        ],
        modifier="",
    )
    namespace = Namespace(name="featuretest", body=[StateOptions, State, func])
    includes = [
        "#include <Eigen/Dense>    // Matrix",
        "#include <any>            // any",
        "#include <cstddef>        // size_t",
        "#include <iostream>       // std::cout, debugging",
        "#include <optional>       // optional",
        "#include <unordered_map>  // unordered_map",
    ]
    header = HeaderFile(pragma=True, includes=includes, namespaces=[namespace])
    return header


def source_definition():
    body = [
        ConstructorDefinition("State"),
        ConstructorDefinition(
            "State",
            args=[Arg("const StateOptions&", "options")],
            initializer_list=[
                (
                    "data",
                    ", ".join((f"options.{member}" for member in MEMBERS)),
                )
            ],
        ),
        FunctionDef(
            "State",
            "elementwise_clamp",
            args=[
                Arg("const State&", "value"),
                Arg("const State&", "lower"),
                Arg("const State&", "upper"),
            ],
            modifier="",
            body=[
                Escape("State result;"),
                Escape(
                    "result.CON_pos_pos_x() = std::clamp(value.CON_pos_pos_x(), lower.CON_pos_pos_x(), upper.CON_pos_pos_x());"
                ),
                Escape(
                    "result.CON_pos_pos_y() = std::clamp(value.CON_pos_pos_y(), lower.CON_pos_pos_y(), upper.CON_pos_pos_y());"
                ),
                Escape(
                    "result.CON_pos_pos_z() = std::clamp(value.CON_pos_pos_z(), lower.CON_pos_pos_z(), upper.CON_pos_pos_z());"
                ),
                Return("result"),
            ],
        ),
    ]

    namespace = Namespace(name="featuretest", body=body)
    includes = [
        "#include <example.h>",
    ]
    source = SourceFile(includes=includes, namespaces=[namespace])
    return source


def main(header_location, source_location):
    logging.debug("\n\nDebug Header!\n")

    with open(header_location, "w") as f:
        for idx, line in enumerate(header_definition().compile(CompileState(indent=2))):
            logging.debug(f"{str(idx).rjust(5)} | {line}")
            f.write("%s\n" % line)

    logging.debug("\n\nDebug Source!\n")

    with open(source_location, "w") as f:
        for idx, line in enumerate(source_definition().compile(CompileState(indent=2))):
            logging.debug(f"{str(idx).rjust(5)} | {line}")
            f.write("%s\n" % line)


if __name__ == "__main__":
    import sys

    main(sys.argv[1], sys.argv[2])
