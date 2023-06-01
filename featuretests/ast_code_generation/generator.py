from itertools import chain
from typing import Optional
import re
import difflib

from formak.ast_tools import (
    Arg,
    ClassDef,
    CompileState,
    ConstructorDeclaration,
    EnumClassDef,
    Escape,
    ForwardClassDeclaration,
    FunctionDeclaration,
    FunctionDef,
    HeaderFile,
    If,
    MemberDeclaration,
    Namespace,
    Return,
    Templated,
    UsingDeclaration,
)


def tprint(ast):
    """
    tprint aka test_print

    Incrementally print each line so that logging output will show previous lines. Exceptions will bubble up for the line
    """
    for line in ast.compile(CompileState(indent=2)):
        print(line)


def header_definition():
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
    namespace = Namespace(name="featuretest", body=[StateOptions, State])
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


def main(target_location):
    print("\n\nDebug!\n")
    with open(target_location, "w") as f:
        for line in header_definition().compile(CompileState(indent=2)):
            print(line)
            f.write("%s\n" % line)


if __name__ == "__main__":
    import sys

    main(sys.argv[1])
