from itertools import chain
from typing import Iterable

from formak.ast_tools import (
    Arg,
    BaseAst,
    ClassDef,
    ConstructorDeclaration,
    ConstructorDefinition,
    EnumClassDef,
    FromFileTemplate,
    FunctionDeclaration,
    FunctionDef,
    MemberDeclaration,
    Private,
    Public,
    Return,
    Templated,
    UsingDeclaration,
)


def StateOptions(generator) -> BaseAst:
    return ClassDef(
        "struct",
        "StateOptions",
        bases=[],
        body=[
            MemberDeclaration("double", member, 0.0)
            for member in generator.arglist_state
        ],
    )


def State(generator) -> BaseAst:
    return ClassDef(
        "struct",
        "State",
        bases=[],
        body=[
            MemberDeclaration("static constexpr size_t", "rows", generator.state_size),
            MemberDeclaration("static constexpr size_t", "cols", 1),
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
                    for idx, name in enumerate(generator.arglist_state)
                ]
            )
        )
        + [
            MemberDeclaration("DataT", "data", "DataT::Zero()"),
        ],
    )


def ControlOptions(generator) -> BaseAst:
    return ClassDef(
        "struct",
        "ControlOptions",
        bases=[],
        body=[
            MemberDeclaration("double", member, 0.0)
            for member in generator.arglist_control
        ],
    )


def Control(generator) -> BaseAst:
    return ClassDef(
        "struct",
        "Control",
        bases=[],
        body=[
            MemberDeclaration(
                "static constexpr size_t", "rows", generator.control_size
            ),
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
                    for idx, name in enumerate(generator.arglist_control)
                ]
            )
        )
        + [
            MemberDeclaration("DataT", "data", "DataT::Zero()"),
        ],
    )


def CalibrationOptions(generator) -> BaseAst:
    return ClassDef(
        "struct",
        "CalibrationOptions",
        bases=[],
        body=[
            MemberDeclaration("double", member, 0.0)
            for member in generator.arglist_calibration
        ],
    )


def Calibration(generator) -> BaseAst:
    return ClassDef(
        "struct",
        "Calibration",
        bases=[],
        body=[
            MemberDeclaration(
                "static constexpr size_t", "rows", generator.calibration_size
            ),
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
                    for idx, name in enumerate(generator.arglist_calibration)
                ]
            )
        )
        + [
            MemberDeclaration("DataT", "data", "DataT::Zero()"),
        ],
    )


def standard_process_args(generator) -> Iterable[Arg]:
    # TODO(buck): Would mypy catch the case of yielding
    #       yield Arg("double", "dt"),
    # which yields a tuple and not an Arg (see trailing comma)
    yield Arg("double", "dt")
    if generator.enable_EKF:
        yield Arg("const StateAndVariance&", "state")
    else:
        yield Arg("const State&", "state")

    if generator.enable_calibration():
        yield Arg("const Calibration&", "calibration")
    if generator.enable_control():
        yield Arg("const Control&", "control")


def standard_reading_args(generator, reading_type=None) -> Iterable[Arg]:
    if generator.enable_EKF:
        yield Arg("const StateAndVariance&", "state")
    else:
        yield Arg("const State&", "state")

    if generator.enable_calibration():
        yield Arg("const Calibration&", "calibration")

    if reading_type is None:
        yield Arg("const ReadingT&", "reading")
    else:
        yield Arg(f"const {reading_type.typename}&", "reading")


def State_model(generator) -> BaseAst:
    return FunctionDeclaration(
        "State",
        "model",
        args=standard_process_args(generator),
        modifier="",
    )


def StateAndVariance_process_model(generator) -> BaseAst:
    return FunctionDeclaration(
        "StateAndVariance",
        "process_model",
        args=standard_process_args(generator),
        modifier="const",
    )


def StateAndVariance_sensor_model(generator) -> BaseAst:
    return Templated(
        [Arg("typename", "ReadingT")],
        FunctionDef(
            "StateAndVariance",
            "sensor_model",
            args=standard_reading_args(generator),
            modifier="const",
            body=[
                FromFileTemplate(
                    "sensor_model.hpp",
                    inserts={
                        "enable_calibration": generator.enable_calibration(),
                    },
                ),
            ],
        ),
    )


def Covariance(generator) -> BaseAst:
    return ClassDef(
        "struct",
        "Covariance",
        bases=[],
        body=[
            MemberDeclaration("static constexpr size_t", "rows", generator.state_size),
            MemberDeclaration("static constexpr size_t", "cols", generator.state_size),
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
                    for idx, name in enumerate(generator.arglist_state)
                ]
            )
        )
        + [
            MemberDeclaration("DataT", "data", "DataT::Identity()"),
        ],
    )


def StateAndVariance(generator) -> BaseAst:
    return ClassDef(
        "struct",
        "StateAndVariance",
        bases=[],
        body=[
            MemberDeclaration("State", "state"),
            MemberDeclaration("Covariance", "covariance"),
        ],
    )


def SensorId(generator) -> BaseAst:
    return EnumClassDef(
        "SensorId",
        members=[f"{name.upper()}" for name, _, _ in generator.sensorlist],
    )


def _EKF_Tag_body(generator) -> Iterable[BaseAst]:
    yield UsingDeclaration(
        "StateAndVarianceT",
        "StateAndVariance",
    )
    if generator.enable_calibration():
        yield UsingDeclaration(
            "CalibrationT",
            "Calibration",
        )
    else:
        yield UsingDeclaration("CalibrationT", "std::false_type")
    if generator.enable_control():
        yield UsingDeclaration(
            "ControlT",
            "Control",
        )
    else:
        yield UsingDeclaration("ControlT", "std::false_type")
    yield UsingDeclaration(
        "StampedReadingBaseT",
        "StampedReadingBase",
    )
    yield MemberDeclaration(
        "static constexpr double", "max_dt_sec", "cpp::Config::max_dt_sec"
    )


def EKF_Tag(generator) -> BaseAst:
    return ClassDef("struct", "Tag", bases=[], body=_EKF_Tag_body(generator))


# TODO(buck): I can make this name "nice" again as just ExtendedKalmanFilter if it has the prefix of a helper file (e.g. fragments.ExtendedKalmanFilter)
def ExtendedKalmanFilter_ccode(generator) -> BaseAst:
    return ClassDef(
        "class",
        "ExtendedKalmanFilter",
        bases=[],
        body=[
            Public(),
            EKF_Tag(generator),
            UsingDeclaration(
                "CovarianceT",
                f"Eigen::Matrix<double, {generator.control_size}, {generator.control_size}>",
            ),
            UsingDeclaration(
                "ProcessJacobianT",
                f"Eigen::Matrix<double, {generator.state_size}, {generator.state_size}>",
            ),
            UsingDeclaration(
                "ControlJacobianT",
                f"Eigen::Matrix<double, {generator.state_size}, {generator.control_size}>",
            ),
            UsingDeclaration("ProcessModel", "ExtendedKalmanFilterProcessModel"),
            StateAndVariance_process_model(generator),
            StateAndVariance_sensor_model(generator),
            Templated(
                [Arg("typename", "ReadingT")],
                FunctionDef(
                    "std::optional<typename ReadingT::InnovationT>",
                    "innovations",
                    args=[],
                    modifier="",
                    body=[
                        FromFileTemplate(
                            "innovations.hpp",
                            inserts={},
                        )
                    ],
                ),
            ),
            Private(),
            MemberDeclaration(
                "mutable std::unordered_map<SensorId, std::any>", "_innovations"
            ),
        ],
    )


def ExtendedKalmanFilterProcessModel_model(generator) -> BaseAst:
    return FunctionDeclaration(
        "static State", "model", args=standard_process_args(generator), modifier=""
    )


def ExtendedKalmanFilterProcessModel_process_jacobian(generator) -> BaseAst:
    return FunctionDeclaration(
        "static typename ExtendedKalmanFilter::ProcessJacobianT",
        "process_jacobian",
        args=standard_process_args(generator),
        modifier="",
    )


def ExtendedKalmanFilterProcessModel_control_jacobian(generator) -> BaseAst:
    return FunctionDeclaration(
        "static typename ExtendedKalmanFilter::ControlJacobianT",
        "control_jacobian",
        args=standard_process_args(generator),
        modifier="",
    )


def ExtendedKalmanFilterProcessModel_covariance(generator) -> BaseAst:
    return FunctionDeclaration(
        "static typename ExtendedKalmanFilter::CovarianceT",
        "covariance",
        args=standard_process_args(generator),
        modifier="",
    )


def ExtendedKalmanFilterProcessModel(generator) -> BaseAst:
    return ClassDef(
        "class",
        "ExtendedKalmanFilterProcessModel",
        bases=[],
        body=[
            Public(),
            ExtendedKalmanFilterProcessModel_model(generator),
            ExtendedKalmanFilterProcessModel_process_jacobian(generator),
            ExtendedKalmanFilterProcessModel_control_jacobian(generator),
            ExtendedKalmanFilterProcessModel_covariance(generator),
        ],
    )


def _StampedReadingBase_args(generator) -> Iterable[BaseAst]:
    yield Arg("const ExtendedKalmanFilter&", "impl")
    yield Arg("const StateAndVariance&", "state")
    if generator.enable_calibration():
        yield Arg("const Calibration&", "calibration")


def StampedReadingBase(generator) -> BaseAst:
    return ClassDef(
        "struct",
        "StampedReadingBase",
        bases=[],
        body=[
            FunctionDeclaration(
                "virtual StateAndVariance",
                "sensor_model",
                args=_StampedReadingBase_args(generator),
                modifier="const = 0",
            ),
        ],
    )


def ReadingOptions(reading_type) -> BaseAst:
    return ClassDef(
        "struct",
        f"{reading_type.typename}Options",
        bases=[],
        body=[
            MemberDeclaration("double", symbol, 0.0)
            for symbol in sorted(list(reading_type.sensor_model_mapping.keys()))
        ],
    )


def _Reading_sensor_model_body(generator) -> Iterable[BaseAst]:
    if generator.enable_calibration():
        yield Return("impl.sensor_model(state, calibration, *this)")
    else:
        yield Return("impl.sensor_model(state, *this)")


def _Reading_sensor_model_args(generator) -> Iterable[Arg]:
    yield Arg("const ExtendedKalmanFilter&", "impl")
    yield Arg("const StateAndVariance&", "state")
    if generator.enable_calibration():
        yield Arg("const Calibration&", "calibration")


def _Reading_sensor_model_function_def(generator) -> BaseAst:
    return FunctionDef(
        "StateAndVariance",
        "sensor_model",
        args=_Reading_sensor_model_args(generator),
        modifier="const override",
        body=_Reading_sensor_model_body(generator),
    )


def Reading(generator, reading_type) -> BaseAst:
    return ClassDef(
        "struct",
        f"{reading_type.typename}",
        bases=["StampedReadingBase"],
        body=[
            UsingDeclaration("DataT", f"Eigen::Matrix<double, {reading_type.size}, 1>"),
            UsingDeclaration(
                "CovarianceT",
                f"Eigen::Matrix<double, {reading_type.size}, {reading_type.size}>",
            ),
            UsingDeclaration(
                "InnovationT",
                f"Eigen::Matrix<double, {reading_type.size}, 1>",
            ),
            UsingDeclaration(
                "KalmanGainT",
                f"Eigen::Matrix<double, {generator.state_size}, {reading_type.size}>",
            ),
            UsingDeclaration(
                "SensorJacobianT",
                f"Eigen::Matrix<double, {reading_type.size}, {generator.state_size}>",
            ),
            UsingDeclaration("SensorModel", f"{reading_type.typename}SensorModel"),
            ConstructorDeclaration(args=[]),
            ConstructorDeclaration(
                args=[Arg(f"const {reading_type.typename}Options&", "options")]
            ),
            _Reading_sensor_model_function_def(generator),
        ]
        + [
            FunctionDef(
                "double",
                name,
                args=[],
                modifier="",
                body=[Return(f"data({idx}, 0)")],
            )
            for idx, name in enumerate(
                sorted(list(reading_type.sensor_model_mapping.keys()))
            )
        ]
        + [
            FunctionDeclaration(
                f"static {reading_type.typename}",
                "model",
                args=standard_reading_args(generator, reading_type),
                modifier="",
            ),
            FunctionDeclaration(
                f"static {reading_type.typename}::SensorJacobianT",
                "jacobian",
                args=standard_reading_args(generator, reading_type),
                modifier="",
            ),
            FunctionDeclaration(
                f"static {reading_type.typename}::CovarianceT",
                "covariance",
                args=standard_reading_args(generator, reading_type),
                modifier="",
            ),
            MemberDeclaration("DataT", "data", "DataT::Zero()"),
            MemberDeclaration("constexpr static size_t", "size", reading_type.size),
            MemberDeclaration(
                "constexpr static SensorId",
                "Identifier",
                reading_type.identifier,
            ),
        ],
    )


def ReadingSensorModel(generator, reading_type) -> BaseAst:
    return ClassDef(
        "struct",
        f"{reading_type.typename}SensorModel",
        bases=[],
        body=[
            FunctionDeclaration(
                f"static {reading_type.typename}",
                "model",
                args=standard_reading_args(generator, reading_type),
                modifier="",
            ),
            FunctionDeclaration(
                f"static {reading_type.typename}::SensorJacobianT",
                "jacobian",
                args=standard_reading_args(generator, reading_type),
                modifier="",
            ),
            FunctionDeclaration(
                f"static {reading_type.typename}::CovarianceT",
                "covariance",
                args=standard_reading_args(generator, reading_type),
                modifier="",
            ),
        ],
    )


def Model_ccode(generator) -> BaseAst:
    return ClassDef(
        "class",
        "Model",
        bases=[],
        body=[
            Public(),
            State_model(
                generator,
            ),
        ],
    )


def StateDefaultConstructor() -> BaseAst:
    return ConstructorDefinition(
        "State", args=[], initializer_list=[("data", "DataT::Zero()")]
    )


def StateOptionsConstructor(generator) -> BaseAst:
    return ConstructorDefinition(
        "State",
        args=[Arg("const StateOptions&", "options")],
        initializer_list=[
            (
                "data",
                ", ".join(f"options.{name}" for name in generator.arglist_state),
            ),
        ],
    )


def CalibrationDefaultConstructor() -> BaseAst:
    return ConstructorDefinition(
        "Calibration", args=[], initializer_list=[("data", "DataT::Zero()")]
    )


def CalibrationConstructor(generator) -> BaseAst:
    return ConstructorDefinition(
        "Calibration",
        args=[Arg("const CalibrationOptions&", "options")],
        initializer_list=[
            (
                "data",
                ", ".join(f"options.{name}" for name in generator.arglist_calibration),
            ),
        ],
    )


def ControlDefaultConstructor() -> BaseAst:
    return ConstructorDefinition(
        "Control", args=[], initializer_list=[("data", "DataT::Zero()")]
    )


def ControlConstructor(generator) -> BaseAst:
    return ConstructorDefinition(
        "Control",
        args=[Arg("const ControlOptions&", "options")],
        initializer_list=[
            (
                "data",
                ", ".join(f"options.{name}" for name in generator.arglist_control),
            ),
        ],
    )


def EKF_process_model(generator) -> BaseAst:
    return FunctionDef(
        "StateAndVariance",
        "ExtendedKalmanFilter::process_model",
        modifier="const",
        args=standard_process_args(generator),
        body=[
            FromFileTemplate(
                "process_model.cpp",
                inserts={
                    "enable_control": generator.enable_control(),
                    "enable_calibration": generator.enable_calibration(),
                },
            ),
        ],
    )


def EKFPM_model(generator) -> BaseAst:
    return FunctionDef(
        "State",
        "ExtendedKalmanFilterProcessModel::model",
        modifier="",
        args=standard_process_args(generator),
        body=generator.process_model_body(),
    )


def EKFPM_process_jacobian(generator) -> BaseAst:
    return FunctionDef(
        "typename ExtendedKalmanFilter::ProcessJacobianT",
        "ExtendedKalmanFilterProcessModel::process_jacobian",
        modifier="",
        args=standard_process_args(generator),
        body=generator.process_jacobian_body(),
    )


def EKFPM_control_jacobian(generator) -> BaseAst:
    return FunctionDef(
        "typename ExtendedKalmanFilter::ControlJacobianT",
        "ExtendedKalmanFilterProcessModel::control_jacobian",
        modifier="",
        args=standard_process_args(generator),
        body=generator.control_jacobian_body(),
    )


def EKFPM_covariance(generator) -> BaseAst:
    return FunctionDef(
        "typename ExtendedKalmanFilter::CovarianceT",
        "ExtendedKalmanFilterProcessModel::covariance",
        modifier="",
        args=standard_process_args(generator),
        body=generator.control_covariance_body(),
    )


def ReadingDefaultConstructor(reading_type) -> BaseAst:
    return ConstructorDefinition(
        reading_type.typename,
        args=[],
        initializer_list=[("data", "DataT::Zero()")],
    )


def ReadingConstructor(reading_type) -> BaseAst:
    return ConstructorDefinition(
        reading_type.typename,
        args=[Arg(f"const {reading_type.typename}Options&", "options")],
        initializer_list=[
            (
                "data",
                ", ".join(
                    f"options.{name}"
                    for name in sorted(list(reading_type.sensor_model_mapping.keys()))
                ),
            )
        ],
    )


def ReadingSensorModel_model(generator, reading_type) -> BaseAst:
    return FunctionDef(
        reading_type.typename,
        f"{reading_type.typename}SensorModel::model",
        modifier="",
        args=standard_reading_args(generator, reading_type),
        body=reading_type.SensorModel_model_body,
    )


def ReadingSensorModel_covariance(generator, reading_type) -> BaseAst:
    return FunctionDef(
        f"{reading_type.typename}::CovarianceT",
        f"{reading_type.typename}SensorModel::covariance",
        modifier="",
        args=standard_reading_args(generator, reading_type),
        body=reading_type.SensorModel_covariance_body,
    )


def ReadingSensorModel_jacobian(generator, reading_type) -> BaseAst:
    return FunctionDef(
        f"{reading_type.typename}::SensorJacobianT",
        f"{reading_type.typename}SensorModel::jacobian",
        modifier="",
        args=standard_reading_args(generator, reading_type),
        body=reading_type.SensorModel_jacobian_body,
    )


def Model_model(generator) -> BaseAst:
    return FunctionDef(
        "State",
        "Model::model",
        modifier="",
        args=standard_process_args(generator),
        body=generator.model_body(),
    )
