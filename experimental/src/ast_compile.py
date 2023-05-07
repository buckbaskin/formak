import ast
from itertools import chain
from typing import List, Any, Optional
from dataclasses import dataclass

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

  struct Covariance {
    using DataT = Eigen::Matrix<double, 9, 9>;


    double& CON_ori_pitch() { return data(0, 0); }
    double CON_ori_pitch() const { return data(0, 0); }
    double& CON_ori_roll() { return data(1, 1); }
    double CON_ori_roll() const { return data(1, 1); }
    double& CON_ori_yaw() { return data(2, 2); }
    double CON_ori_yaw() const { return data(2, 2); }
    double& CON_pos_pos_x() { return data(3, 3); }
    double CON_pos_pos_x() const { return data(3, 3); }
    double& CON_pos_pos_y() { return data(4, 4); }
    double CON_pos_pos_y() const { return data(4, 4); }
    double& CON_pos_pos_z() { return data(5, 5); }
    double CON_pos_pos_z() const { return data(5, 5); }
    double& CON_vel_x() { return data(6, 6); }
    double CON_vel_x() const { return data(6, 6); }
    double& CON_vel_y() { return data(7, 7); }
    double CON_vel_y() const { return data(7, 7); }
    double& CON_vel_z() { return data(8, 8); }
    double CON_vel_z() const { return data(8, 8); }

    DataT data = DataT::Identity();
  };




  struct ControlOptions {

    double IMU_reading_acc_x = 0.0;
    double IMU_reading_acc_y = 0.0;
    double IMU_reading_acc_z = 0.0;
    double IMU_reading_pitch_rate = 0.0;
    double IMU_reading_roll_rate = 0.0;
    double IMU_reading_yaw_rate = 0.0;

  };

  struct Control {
    Control();
    Control(const ControlOptions& options);

    double& IMU_reading_acc_x() {return data(0, 0); }
    double IMU_reading_acc_x() const {return data(0, 0); }
    double& IMU_reading_acc_y() {return data(1, 0); }
    double IMU_reading_acc_y() const {return data(1, 0); }
    double& IMU_reading_acc_z() {return data(2, 0); }
    double IMU_reading_acc_z() const {return data(2, 0); }
    double& IMU_reading_pitch_rate() {return data(3, 0); }
    double IMU_reading_pitch_rate() const {return data(3, 0); }
    double& IMU_reading_roll_rate() {return data(4, 0); }
    double IMU_reading_roll_rate() const {return data(4, 0); }
    double& IMU_reading_yaw_rate() {return data(5, 0); }
    double IMU_reading_yaw_rate() const {return data(5, 0); }

    Eigen::Matrix<double, 6, 1> data =
        Eigen::Matrix<double, 6, 1>::Zero();
  };

  struct CalibrationOptions {

    double IMU_ori_pitch = 0.0;
    double IMU_ori_roll = 0.0;
    double IMU_ori_yaw = 0.0;
    double IMU_pos_x = 0.0;
    double IMU_pos_y = 0.0;
    double IMU_pos_z = 0.0;

  };

  struct Calibration {
    Calibration();
    Calibration(const CalibrationOptions& options);

    double& IMU_ori_pitch() {return data(0, 0); }
    double IMU_ori_pitch() const {return data(0, 0); }
    double& IMU_ori_roll() {return data(1, 0); }
    double IMU_ori_roll() const {return data(1, 0); }
    double& IMU_ori_yaw() {return data(2, 0); }
    double IMU_ori_yaw() const {return data(2, 0); }
    double& IMU_pos_x() {return data(3, 0); }
    double IMU_pos_x() const {return data(3, 0); }
    double& IMU_pos_y() {return data(4, 0); }
    double IMU_pos_y() const {return data(4, 0); }
    double& IMU_pos_z() {return data(5, 0); }
    double IMU_pos_z() const {return data(5, 0); }

    Eigen::Matrix<double, 6, 1> data =
        Eigen::Matrix<double, 6, 1>::Zero();
  };

  struct StateAndVariance {
    State state;
    Covariance covariance;
  };

  enum class SensorId {
    ALTITUDE
  };

  // ReadingTSensorModel
  struct AltitudeSensorModel;

  struct AltitudeOptions {
    double altitude = 0.0;
  };

  // ReadingT
  struct Altitude {
    using DataT = Eigen::Matrix<double, 1, 1>;
    using CovarianceT = Eigen::Matrix<double, 1, 1>;
    using InnovationT = Eigen::Matrix<double, 1, 1>;
    using KalmanGainT = Eigen::Matrix<double, 9, 1>;
    using SensorJacobianT = Eigen::Matrix<double, 1, 9>;
    using SensorModel = AltitudeSensorModel;

    Altitude();
    Altitude(const AltitudeOptions& options);

    double& altitude() { return data(0, 0); }

    DataT data = DataT::Zero();

    constexpr static size_t size = 1;
    constexpr static SensorId Identifier = SensorId::ALTITUDE;
  };

  std::ostream& operator<<(std::ostream& o, const Altitude& reading) {
      o << "Reading(data[1, 1] = " << reading.data << ")";
      return o;
  }

  struct AltitudeSensorModel {
      static Altitude model(
        const StateAndVariance& input,
        const Calibration& input_calibration,
        const Altitude& input_reading);

    static typename Altitude
    ::SensorJacobianT jacobian(
        const StateAndVariance& input,
        const Calibration& input_calibration,
        const Altitude& input_reading);

    static typename Altitude
    ::CovarianceT covariance(
        const StateAndVariance& input,
        const Calibration& input_calibration,
        const Altitude& input_reading);
  };

  class ExtendedKalmanFilterProcessModel;

  class ExtendedKalmanFilter {
   public:
    using CovarianceT =
        Eigen::Matrix<double, 6, 6>;
    using ProcessJacobianT =
        Eigen::Matrix<double, 9, 9>;
    using ControlJacobianT =
        Eigen::Matrix<double, 9, 6>;
    using ProcessModel = ExtendedKalmanFilterProcessModel;

    StateAndVariance process_model(
        double dt,
        const StateAndVariance& input
        ,
        const Calibration& input_calibration
        ,
        const Control& input_control
    );

    template <typename ReadingT>
    StateAndVariance sensor_model(
        const StateAndVariance& input,
        const Calibration& input_calibration,
        const ReadingT& input_reading) {
      const State& state = input.state;                 // mu
      const Covariance& covariance = input.covariance;  // Sigma

      // z_est = sensor_model()
      const ReadingT reading_est =
          ReadingT::SensorModel::model(input,
                                       input_calibration,
                                       input_reading);  // z_est

      // H = Jacobian(z_est w.r.t. state)
      const typename ReadingT::SensorJacobianT H =
          ReadingT::SensorModel::jacobian(input,
                                          input_calibration,
                                          input_reading);

      // Project State Noise into Sensor Space
      // S = H * Sigma * H.T + Q_t
      const typename ReadingT::CovarianceT sensor_estimate_covariance =
          H * covariance.data * H.transpose() +
          ReadingT::SensorModel::covariance(input,
                                            input_calibration,
                                            input_reading);

      // S_inv = inverse(S)
      const typename ReadingT::CovarianceT S_inv =
          sensor_estimate_covariance.inverse();

      // Kalman Gain
      // K = Sigma * H.T * S_inv
      const typename ReadingT::KalmanGainT kalman_gain =
          covariance.data * H.transpose() * S_inv;

      // Innovation
      // innovation = z - z_est
      const typename ReadingT::InnovationT innovation =
          input_reading.data - reading_est.data;
      _innovations[ReadingT::Identifier] = innovation;

      // Update State Estimate
      // next_state = state + K * innovation
      State next_state;
      next_state.data = state.data + kalman_gain * innovation;

      // Update Covariance
      // next_covariance = Sigma - K * H * Sigma
      Covariance next_covariance;
      next_covariance.data =
          covariance.data - kalman_gain * H * covariance.data;

      // TODO(buck): Measurement Likelihood (optional)

      // Here be the StateAndVariance math
      return StateAndVariance{.state = next_state,
                              .covariance = next_covariance};
    }

    template <typename ReadingT>
    std::optional<typename ReadingT::InnovationT> innovations() {
      if (_innovations.count(ReadingT::Identifier) > 0) {
        return std::any_cast<typename ReadingT::InnovationT>(
            _innovations[ReadingT::Identifier]);
      }
      return {};
    }

   private:
    std::unordered_map<SensorId, std::any> _innovations;
  };

  class ExtendedKalmanFilterProcessModel {
   public:
    static State model(
        double dt,
        const StateAndVariance& input
        ,
        const Calibration& input_calibration
        ,
        const Control& input_control
    );

    static typename ExtendedKalmanFilter::ProcessJacobianT process_jacobian(
        double dt,
        const StateAndVariance& input
        ,
        const Calibration& input_calibration
        ,
        const Control& input_control
    );

    static typename ExtendedKalmanFilter::ControlJacobianT control_jacobian(
        double dt,
        const StateAndVariance& input
        ,
        const Calibration& input_calibration
        ,
        const Control& input_control
    );

    static typename ExtendedKalmanFilter::CovarianceT covariance(
        double dt,
        const StateAndVariance& input,
        const Calibration& input_calibration      ,
        const Control& input_control
    );
  };

}  // namespace featuretest
"""

# comments optional
# also skip all the formatting (if I roll my own?)

# Things to represent
#   - forward declaration
#   - class definition
#   - pragma once for header
#   - include list


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

        yield "}"


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
        yield f"{classname}("
        for arg in self.args:
            for line in arg.compile(options, classname=classname, **kwargs):
                yield line + ","
        yield ");"


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
                    # TODO(buck): fill in the body
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

print("\n".join(header.compile(CompileState(indent=2))))
