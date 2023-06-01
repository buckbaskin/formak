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


def classdef_State():
    """
    struct State {
      static constexpr size_t rows = 9;
      static constexpr size_t cols = 1;
      using DataT = Eigen::Matrix<double, rows, cols>;

      State();
      State(const StateOptions& options);

      double& CON_ori_pitch() {
        return data(0, 0);
      }
      double CON_ori_pitch() const {
        return data(0, 0);
      }
      double& CON_ori_roll() {
        return data(1, 0);
      }
      double CON_ori_roll() const {
        return data(1, 0);
      }
      double& CON_ori_yaw() {
        return data(2, 0);
      }
      double CON_ori_yaw() const {
        return data(2, 0);
      }
      double& CON_pos_pos_x() {
        return data(3, 0);
      }
      double CON_pos_pos_x() const {
        return data(3, 0);
      }
      double& CON_pos_pos_y() {
        return data(4, 0);
      }
      double CON_pos_pos_y() const {
        return data(4, 0);
      }
      double& CON_pos_pos_z() {
        return data(5, 0);
      }
      double CON_pos_pos_z() const {
        return data(5, 0);
      }
      double& CON_vel_x() {
        return data(6, 0);
      }
      double CON_vel_x() const {
        return data(6, 0);
      }
      double& CON_vel_y() {
        return data(7, 0);
      }
      double CON_vel_y() const {
        return data(7, 0);
      }
      double& CON_vel_z() {
        return data(8, 0);
      }
      double CON_vel_z() const {
        return data(8, 0);
      }

      DataT data = DataT::Zero();
    };
    """
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
    namespace = Namespace(name="featuretest", body=[State])
    return namespace.compile(CompileState())


def main(target_location):
    print("\n\nDebug!\n")
    with open(target_location, 'w') as f:
        for line in classdef_State():
            print(line)
            f.write('%s\n' % line)


if __name__ == "__main__":
    import sys

    main(sys.argv[1])
