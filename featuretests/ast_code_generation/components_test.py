from itertools import chain
import difflib

from formak.ast_tools import (
    CompileState,
    Namespace,
    HeaderFile,
    ClassDef,
    MemberDeclaration,
    UsingDeclaration,
    ConstructorDeclaration,
    Arg,
    FunctionDef,
    Return,
)


def tprint(ast):
    """
    tprint aka test_print

    Incrementally print each line so that logging output will show previous lines. Exceptions will bubble up for the line
    """
    for line in ast.compile(CompileState(indent=2)):
        print(line)


def generator_compare(test):
    def wrapped(*args, **kwargs):
        if test.__doc__ is None or test.__doc__ == "":
            raise ValueError(
                f"Test {test.__name__} is missing docstring to compare with generator_compare"
            )

        expected = list(
            filter(
                lambda s: len(s) > 0,
                map(lambda s: s.strip(), test.__doc__.splitlines(keepends=False)),
            )
        )

        result = list(
            filter(
                lambda s: len(s) > 0,
                map(
                    lambda s: s.strip(),
                    test(*args, **kwargs).compile(CompileState(indent=2)),
                ),
            )
        )

        diff = difflib.ndiff(expected, result, linejunk=difflib.IS_LINE_JUNK)
        print(f"diff {type(diff)}")
        for idx, line in enumerate(diff):
            print(str(idx).rjust(3), line)

        1 / 0

    # TODO(buck): annotate the wrapper

    return wrapped


gen_comp = generator_compare


@gen_comp
def test_classdef_stateoptions():
    """
    struct StateOptions {

      double CON_ori_pitch = 0.0;
      double CON_ori_roll = 0.0;
      double CON_ori_yaw = 0.0;
      double CON_pos_pos_x = 0.0;
      double CON_pos_pos_y = 0.0;
      double CON_pos_pos_z = 0.0;
      double CON_vel_x = 0.0;
      double CON_vel_y = 0.0;
      double CON_vel_z = 0.0;

    };
    """
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
    return StateOptions


@gen_comp
def test_classdef_state():
    """
    struct State {
      static constexpr size_t rows = 9;
      static constexpr size_t cols = 1;
      using DataT = Eigen::Matrix<double, rows, cols>;

      State();
      State(const StateOptions& options);

      double& CON_ori_pitch() { return data(0, 0); }
      double CON_ori_pitch() const { return data(0, 0); }
      double& CON_ori_roll() { return data(1, 0); }
      double CON_ori_roll() const { return data(1, 0); }
      double& CON_ori_yaw() { return data(2, 0); }
      double CON_ori_yaw() const { return data(2, 0); }
      double& CON_pos_pos_x() { return data(3, 0); }
      double CON_pos_pos_x() const { return data(3, 0); }
      double& CON_pos_pos_y() { return data(4, 0); }
      double CON_pos_pos_y() const { return data(4, 0); }
      double& CON_pos_pos_z() { return data(5, 0); }
      double CON_pos_pos_z() const { return data(5, 0); }
      double& CON_vel_x() { return data(6, 0); }
      double CON_vel_x() const { return data(6, 0); }
      double& CON_vel_y() { return data(7, 0); }
      double CON_vel_y() const { return data(7, 0); }
      double& CON_vel_z() { return data(8, 0); }
      double CON_vel_z() const { return data(8, 0); }

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
    return State


def test_classdef_covariance():
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
    tprint(Covariance)


def test_classdef_controloptions():
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
    tprint(ControlOptions)


def test_classdef_control():
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
    tprint(Control)


def test_namespace():
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
    namespace = Namespace(name="featuretest", body=[StateOptions])
    tprint(namespace)


def test_headerfile():
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
    namespace = Namespace(name="featuretest", body=[StateOptions])
    includes = [
        "#include <Eigen/Dense>    // Matrix",
        "#include <any>            // any",
        "#include <cstddef>        // size_t",
        "#include <iostream>       // std::cout, debugging",
        "#include <optional>       // optional",
        "#include <unordered_map>  // unordered_map",
    ]
    header = HeaderFile(pragma=True, includes=includes, namespaces=[namespace])

    tprint(header)
