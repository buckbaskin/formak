import difflib
import re
from itertools import chain
from typing import Optional

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


def simplify(line):
    return re.sub(r"\s+", " ", line.strip())


def diff(test, *args, **kwargs) -> Optional[int]:
    """
    Split out logic from generator_compare to simplify what gets rendered in the pytest diff

    Returns Optional[int] containing the line the diff starts if there is a diff
    """
    expected = list(
        filter(
            lambda s: len(s) > 0,
            map(simplify, test.__doc__.splitlines(keepends=False)),
        )
    )

    result = list(
        filter(
            lambda s: len(s) > 0,
            map(
                simplify,
                test(*args, **kwargs).compile(CompileState(indent=2)),
            ),
        )
    )

    diff = difflib.ndiff(expected, result)

    diff_start = None

    print(f"diff {type(diff)}")
    for idx, line in enumerate(diff):
        print(str(idx).rjust(3), line)
        if diff_start is None and line[0] in ["+", "?", "-"]:
            diff_start = idx
        if diff_start is not None and idx >= diff_start + 10:
            print("...", "Trimming diff output")
            break

    return diff_start


def generator_compare(test):
    def wrapped(*args, **kwargs):
        if test.__doc__ is None or test.__doc__ == "":
            raise ValueError(
                f"Test {test.__name__} is missing docstring to compare with generator_compare"
            )

        if diff(test, *args, **kwargs) is not None:
            raise ValueError("Test did not match expected string")

    wrapped.__name__ = test.__name__
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


@gen_comp
def test_classdef_covariance():
    """
    struct Covariance {
      static constexpr size_t rows = 9;
      static constexpr size_t cols = 9;
      using DataT = Eigen::Matrix<double, rows, cols>;


      double& CON_ori_pitch() {
        return data(0, 0);
      }
      double CON_ori_pitch() const {
        return data(0, 0);
      }
      double& CON_ori_roll() {
        return data(1, 1);
      }
      double CON_ori_roll() const {
        return data(1, 1);
      }
      double& CON_ori_yaw() {
        return data(2, 2);
      }
      double CON_ori_yaw() const {
        return data(2, 2);
      }
      double& CON_pos_pos_x() {
        return data(3, 3);
      }
      double CON_pos_pos_x() const {
        return data(3, 3);
      }
      double& CON_pos_pos_y() {
        return data(4, 4);
      }
      double CON_pos_pos_y() const {
        return data(4, 4);
      }
      double& CON_pos_pos_z() {
        return data(5, 5);
      }
      double CON_pos_pos_z() const {
        return data(5, 5);
      }
      double& CON_vel_x() {
        return data(6, 6);
      }
      double CON_vel_x() const {
        return data(6, 6);
      }
      double& CON_vel_y() {
        return data(7, 7);
      }
      double CON_vel_y() const {
        return data(7, 7);
      }
      double& CON_vel_z() {
        return data(8, 8);
      }
      double CON_vel_z() const {
        return data(8, 8);
      }

      DataT data = DataT::Identity();
    };
    """
    Covariance = ClassDef(
        "struct",
        "Covariance",
        bases=[],
        body=[
            MemberDeclaration("static constexpr size_t", "rows", 9),
            MemberDeclaration("static constexpr size_t", "cols", 9),
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
    return Covariance


@gen_comp
def test_classdef_controloptions():
    """
    struct ControlOptions {

      double IMU_reading_acc_x = 0.0;
      double IMU_reading_acc_y = 0.0;
      double IMU_reading_acc_z = 0.0;
      double IMU_reading_pitch_rate = 0.0;
      double IMU_reading_roll_rate = 0.0;
      double IMU_reading_yaw_rate = 0.0;

    };
    """
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
    return ControlOptions


@gen_comp
def test_classdef_control():
    """
    struct Control {
      static constexpr size_t rows = 6;
      static constexpr size_t cols = 1;
      using DataT = Eigen::Matrix<double, rows, cols>;

      Control();
      Control(const ControlOptions& options);

      double& IMU_reading_acc_x() {
        return data(0, 0);
      }
      double IMU_reading_acc_x() const {
        return data(0, 0);
      }
      double& IMU_reading_acc_y() {
        return data(1, 0);
      }
      double IMU_reading_acc_y() const {
        return data(1, 0);
      }
      double& IMU_reading_acc_z() {
        return data(2, 0);
      }
      double IMU_reading_acc_z() const {
        return data(2, 0);
      }
      double& IMU_reading_pitch_rate() {
        return data(3, 0);
      }
      double IMU_reading_pitch_rate() const {
        return data(3, 0);
      }
      double& IMU_reading_roll_rate() {
        return data(4, 0);
      }
      double IMU_reading_roll_rate() const {
        return data(4, 0);
      }
      double& IMU_reading_yaw_rate() {
        return data(5, 0);
      }
      double IMU_reading_yaw_rate() const {
        return data(5, 0);
      }

      DataT data = DataT::Zero();
    };
    """
    Control = ClassDef(
        "struct",
        "Control",
        bases=[],
        body=[
            MemberDeclaration("static constexpr size_t", "rows", 6),
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
    return Control


@gen_comp
def test_classdef_calibrationoptions():
    """
    struct CalibrationOptions {

      double IMU_ori_pitch = 0.0;
      double IMU_ori_roll = 0.0;
      double IMU_ori_yaw = 0.0;
      double IMU_pos_x = 0.0;
      double IMU_pos_y = 0.0;
      double IMU_pos_z = 0.0;

    };

    """
    CalibrationOptions = ClassDef(
        "struct",
        "CalibrationOptions",
        bases=[],
        body=[
            MemberDeclaration("double", "IMU_ori_pitch", 0.0),
            MemberDeclaration("double", "IMU_ori_roll", 0.0),
            MemberDeclaration("double", "IMU_ori_yaw", 0.0),
            MemberDeclaration("double", "IMU_pos_x", 0.0),
            MemberDeclaration("double", "IMU_pos_y", 0.0),
            MemberDeclaration("double", "IMU_pos_z", 0.0),
        ],
    )
    return CalibrationOptions


@gen_comp
def test_classdef_calibration():
    """
    struct Calibration {
      static constexpr size_t rows = 6;
      static constexpr size_t cols = 1;
      using DataT = Eigen::Matrix<double, rows, cols>;

      Calibration();
      Calibration(const CalibrationOptions& options);

      double& IMU_ori_pitch() {
        return data(0, 0);
      }
      double IMU_ori_pitch() const {
        return data(0, 0);
      }
      double& IMU_ori_roll() {
        return data(1, 0);
      }
      double IMU_ori_roll() const {
        return data(1, 0);
      }
      double& IMU_ori_yaw() {
        return data(2, 0);
      }
      double IMU_ori_yaw() const {
        return data(2, 0);
      }
      double& IMU_pos_x() {
        return data(3, 0);
      }
      double IMU_pos_x() const {
        return data(3, 0);
      }
      double& IMU_pos_y() {
        return data(4, 0);
      }
      double IMU_pos_y() const {
        return data(4, 0);
      }
      double& IMU_pos_z() {
        return data(5, 0);
      }
      double IMU_pos_z() const {
        return data(5, 0);
      }

      DataT data = DataT::Zero();
    };

    """
    Calibration = ClassDef(
        "struct",
        "Calibration",
        bases=[],
        body=[
            MemberDeclaration("static constexpr size_t", "rows", 6),
            MemberDeclaration("static constexpr size_t", "cols", 1),
            UsingDeclaration("DataT", "Eigen::Matrix<double, rows, cols>"),
            ConstructorDeclaration(),  # No args constructor gets default constructor
            ConstructorDeclaration(args=[Arg("const CalibrationOptions&", "options")]),
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
                            "IMU_ori_pitch",
                            "IMU_ori_roll",
                            "IMU_ori_yaw",
                            "IMU_pos_x",
                            "IMU_pos_y",
                            "IMU_pos_z",
                        ]
                    )
                ]
            )
        )
        + [
            MemberDeclaration("DataT", "data", "DataT::Zero()"),
        ],
    )
    return Calibration


@gen_comp
def test_classdef_stateandvariance():
    """
    struct StateAndVariance {
      State state;
      Covariance covariance;
    };

    """
    StateAndVariance = ClassDef(
        "struct",
        "StateAndVariance",
        bases=[],
        body=[
            MemberDeclaration("State", "state"),
            MemberDeclaration("Covariance", "covariance"),
        ],
    )
    return StateAndVariance


@gen_comp
def test_enumclassdef_sensorid():
    """
    enum class SensorId {
      ALTITUDE,
    };
    """
    SensorId = EnumClassDef(
        "SensorId",
        members=["ALTITUDE"],
    )
    return SensorId


@gen_comp
def test_forwarddeclaration_altitudesensormodel():
    """
    struct AltitudeSensorModel;
    """
    return ForwardClassDeclaration("struct", "AltitudeSensorModel")


@gen_comp
def test_classdef_altitudeoptions():
    """
    struct AltitudeOptions {
      double altitude = 0.0;
    };
    """
    AltitudeOptions = ClassDef(
        "struct",
        "AltitudeOptions",
        bases=[],
        body=[
            MemberDeclaration("double", "altitude", 0.0),
        ],
    )
    return AltitudeOptions


@gen_comp
def test_classdef_altitude():
    """
    struct Altitude {
      using DataT = Eigen::Matrix<double, 1, 1>;
      using CovarianceT = Eigen::Matrix<double, 1, 1>;
      using InnovationT = Eigen::Matrix<double, 1, 1>;
      using KalmanGainT = Eigen::Matrix<double, 9, 1>;
      using SensorJacobianT = Eigen::Matrix<double, 1, 9>;
      using SensorModel = AltitudeSensorModel;

      Altitude();
      Altitude(const AltitudeOptions& options);

      double& altitude() {
        return data(0, 0);
      }
      double altitude() const {
        return data(0, 0);
      }

      DataT data = DataT::Zero();

      constexpr static size_t size = 1;
      constexpr static SensorId Identifier = SensorId::ALTITUDE;
    };
    """
    Altitude = ClassDef(
        "struct",
        "Altitude",
        bases=[],
        body=[
            UsingDeclaration("DataT", "Eigen::Matrix<double, 1, 1>"),
            UsingDeclaration("CovarianceT", "Eigen::Matrix<double, 1, 1>"),
            UsingDeclaration("InnovationT", "Eigen::Matrix<double, 1, 1>"),
            UsingDeclaration("KalmanGainT", "Eigen::Matrix<double, 9, 1>"),
            UsingDeclaration("SensorJacobianT", "Eigen::Matrix<double, 1, 9>"),
            UsingDeclaration("SensorModel", "AltitudeSensorModel"),
            ConstructorDeclaration(),  # No args constructor gets default constructor
            ConstructorDeclaration(args=[Arg("const AltitudeOptions&", "options")]),
            FunctionDef(
                "double&",
                "altitude",
                args=[],
                modifier="",
                body=[
                    Return("data(0, 0)"),
                ],
            ),
            FunctionDef(
                "double",
                "altitude",
                args=[],
                modifier="const",
                body=[
                    Return("data(0, 0)"),
                ],
            ),
            MemberDeclaration("DataT", "data", "DataT::Zero()"),
            MemberDeclaration("constexpr static size_t", "size", 1),
            MemberDeclaration(
                "constexpr static SensorId", "Identifier", "SensorId::ALTITUDE"
            ),
        ],
    )
    return Altitude


@gen_comp
def test_function_operatorltlt():
    """
    std::ostream& operator<<(
      std::ostream& o,
      const Altitude& reading
    ) {
        o << "Reading(data[1, 1] = " << reading.data << ")";
        return o;
    }
    """
    return FunctionDef(
        "std::ostream&",
        "operator<<",
        args=[Arg("std::ostream&", "o"), Arg("const Altitude&", "reading")],
        modifier="",
        body=[
            Escape('o << "Reading(data[1, 1] = " << reading.data << ")";'),
            Return("o"),
        ],
    )


@gen_comp
def test_classdef_altitudesensormodel():
    """
    struct AltitudeSensorModel {
        static Altitude model(
          const StateAndVariance& input,
          const Calibration& input_calibration,
          const Altitude& input_reading
        );

      static typename Altitude::SensorJacobianT jacobian(
          const StateAndVariance& input,
          const Calibration& input_calibration,
          const Altitude& input_reading
        );

      static typename Altitude::CovarianceT covariance(
          const StateAndVariance& input,
          const Calibration& input_calibration,
          const Altitude& input_reading
          );
    };"""
    return ClassDef(
        "struct",
        "AltitudeSensorModel",
        bases=[],
        body=[
            FunctionDeclaration(
                "static Altitude",
                "model",
                args=[
                    Arg("const StateAndVariance&", "input"),
                    Arg("const Calibration&", "input_calibration"),
                    Arg("const Altitude&", "input_reading"),
                ],
                modifier="",
            ),
            FunctionDeclaration(
                "static typename Altitude::SensorJacobianT",
                "jacobian",
                args=[
                    Arg("const StateAndVariance&", "input"),
                    Arg("const Calibration&", "input_calibration"),
                    Arg("const Altitude&", "input_reading"),
                ],
                modifier="",
            ),
            FunctionDeclaration(
                "static typename Altitude::CovarianceT",
                "covariance",
                args=[
                    Arg("const StateAndVariance&", "input"),
                    Arg("const Calibration&", "input_calibration"),
                    Arg("const Altitude&", "input_reading"),
                ],
                modifier="",
            ),
        ],
    )


@gen_comp
def test_forwarddeclaration_extendedkalmanfilterprocessmodel():
    """
    class ExtendedKalmanFilterProcessModel;
    """
    return ForwardClassDeclaration("class", "ExtendedKalmanFilterProcessModel")


@gen_comp
def test_classdef_extendedkalmanfilter():
    """
    class ExtendedKalmanFilter {
     public:
      using CovarianceT = Eigen::Matrix<double, 6, 6>;
      using ProcessJacobianT = Eigen::Matrix<double, 9, 9>;
      using ControlJacobianT = Eigen::Matrix<double, 9, 6>;
      using ProcessModel = ExtendedKalmanFilterProcessModel;

      StateAndVariance process_model(
          double dt,
          const StateAndVariance& input,
          const Calibration& input_calibration,
          const Control& input_control
      );

      template <typename ReadingT>
      StateAndVariance sensor_model(
          const StateAndVariance& input,
          const Calibration& input_calibration,
          const ReadingT& input_reading
          ) {
        const State& next_state = input.state;                 // mu
        const Covariance& next_covariance = input.covariance;  // Sigma
        // ...
        return StateAndVariance{.state = next_state, .covariance = next_covariance};
      }

      template <typename ReadingT>
      std::optional<typename ReadingT::InnovationT> innovations() {
        if (_innovations.count(ReadingT::Identifier) > 0) {
          return std::any_cast<typename ReadingT::InnovationT>(_innovations[ReadingT::Identifier]);
        }
        return {};
      }

     private:
      std::unordered_map<SensorId, std::any> _innovations;
    };
    """
    return ClassDef(
        "class",
        "ExtendedKalmanFilter",
        [],
        [
            Escape("public:"),
            UsingDeclaration("CovarianceT ", "Eigen::Matrix<double, 6, 6>"),
            UsingDeclaration("ProcessJacobianT ", "Eigen::Matrix<double, 9, 9>"),
            UsingDeclaration("ControlJacobianT ", "Eigen::Matrix<double, 9, 6>"),
            UsingDeclaration("ProcessModel ", "ExtendedKalmanFilterProcessModel"),
            FunctionDeclaration(
                "StateAndVariance",
                "process_model",
                args=[
                    Arg("double", "dt"),
                    Arg("const StateAndVariance&", "input"),
                    Arg("const Calibration&", "input_calibration"),
                    Arg("const Control&", "input_control"),
                ],
                modifier="",
            ),
            Templated(
                [Arg("typename", "ReadingT")],
                FunctionDef(
                    "StateAndVariance",
                    "sensor_model",
                    args=[
                        Arg("const StateAndVariance&", "input"),
                        Arg("const Calibration&", "input_calibration"),
                        Arg("const ReadingT&", "input_reading"),
                    ],
                    modifier="",
                    body=[
                        # Lots of escaping needs work
                        Escape(
                            "const State& next_state = input.state;                 // mu"
                        ),
                        Escape(
                            "const Covariance& next_covariance = input.covariance;  // Sigma"
                        ),
                        Escape("// ..."),
                        Return(
                            "StateAndVariance{.state = next_state, .covariance = next_covariance}"
                        ),
                    ],
                ),
            ),
            Templated(
                [Arg("typename", "ReadingT")],
                FunctionDef(
                    "std::optional<typename ReadingT::InnovationT>",
                    "innovations",
                    [],
                    "",
                    body=[
                        If(
                            "_innovations.count(ReadingT::Identifier) > 0",
                            [
                                Return(
                                    "std::any_cast<typename ReadingT::InnovationT>(_innovations[ReadingT::Identifier])"
                                )
                            ],
                            [],
                        ),
                        Return("{}"),
                    ],
                ),
            ),
            Escape("private:"),
            Escape("std::unordered_map<SensorId, std::any> _innovations;"),
        ],
    )


@gen_comp
def test_classdef_extendedkalmanfilterprocessmodel():
    """
    class ExtendedKalmanFilterProcessModel {
     public:
      static State model(
          double dt,
          const StateAndVariance& input,
          const Calibration& input_calibration,
          const Control& input_control
      );

      static typename ExtendedKalmanFilter::ProcessJacobianT process_jacobian(
          double dt,
          const StateAndVariance& input,
          const Calibration& input_calibration,
          const Control& input_control
      );

      static typename ExtendedKalmanFilter::ControlJacobianT control_jacobian(
          double dt,
          const StateAndVariance& input,
          const Calibration& input_calibration,
          const Control& input_control
      );

      static typename ExtendedKalmanFilter::CovarianceT covariance(
          double dt,
          const StateAndVariance& input,
          const Calibration& input_calibration,
          const Control& input_control
      );
    };
    """
    return ClassDef(
        "class",
        "ExtendedKalmanFilterProcessModel",
        [],
        [
            Escape("public:"),
            FunctionDeclaration(
                "static State",
                "model",
                args=[
                    Arg("double", "dt"),
                    Arg("const StateAndVariance&", "input"),
                    Arg("const Calibration&", "input_calibration"),
                    Arg("const Control&", "input_control"),
                ],
                modifier="",
            ),
            FunctionDeclaration(
                "static typename ExtendedKalmanFilter::ProcessJacobianT",
                "process_jacobian",
                args=[
                    Arg("double", "dt"),
                    Arg("const StateAndVariance&", "input"),
                    Arg("const Calibration&", "input_calibration"),
                    Arg("const Control&", "input_control"),
                ],
                modifier="",
            ),
            FunctionDeclaration(
                "static typename ExtendedKalmanFilter::ControlJacobianT",
                "control_jacobian",
                args=[
                    Arg("double", "dt"),
                    Arg("const StateAndVariance&", "input"),
                    Arg("const Calibration&", "input_calibration"),
                    Arg("const Control&", "input_control"),
                ],
                modifier="",
            ),
            FunctionDeclaration(
                "static typename ExtendedKalmanFilter::CovarianceT",
                "covariance",
                args=[
                    Arg("double", "dt"),
                    Arg("const StateAndVariance&", "input"),
                    Arg("const Calibration&", "input_calibration"),
                    Arg("const Control&", "input_control"),
                ],
                modifier="",
            ),
        ],
    )


@gen_comp
def test_namespace():
    """
    namespace featuretest {
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
    } // namespace featuretest
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
    namespace = Namespace(name="featuretest", body=[StateOptions])
    return namespace


@gen_comp
def test_headerfile():
    """
    #pragma once

    #include <Eigen/Dense>    // Matrix
    #include <any>            // any
    #include <cstddef>        // size_t
    #include <iostream>       // std::cout, debugging
    #include <optional>       // optional
    #include <unordered_map>  // unordered_map


    namespace featuretest {
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
    } // namespace featuretest
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

    return header
