from collections import namedtuple

from backend_cpp.ast import MemberDeclaration, Return
from backend_cpp.basic_block import BasicBlock
from backend_cpp.config import Config
from backend_py.named_covariance import named_covariance
from backend_py.named_vector import named_vector
from formak.exceptions import ModelConstructionError
from sympy import Symbol, diff

# size is the size of the reading for the EKF, not the size of the type
ReadingT = namedtuple(
    "ReadingT",
    [
        "typename",
        "size",
        "identifier",
        "members",
        "initializer_list",
        "Options_members",
        "SensorModel_model_body",
        "SensorModel_covariance_body",
        "SensorModel_jacobian_body",
        "sensor_model_mapping",
    ],
)


class ExtendedKalmanFilter:
    """C++ implementation of the EKF."""

    def __init__(
        self,
        state_model,
        process_noise,
        sensor_models,
        sensor_noises,
        namespace,
        header_include,
        config,
        calibration_map=None,
    ):
        if isinstance(config, dict):
            config = Config(**config)
        assert isinstance(config, Config)
        assert isinstance(process_noise, dict)

        self.enable_EKF = True
        self.config = config
        self.namespace = namespace
        self.header_include = header_include

        # TODO(buck): This is lots of duplication with the model
        self.state_size = len(state_model.state)
        self.calibration_size = len(state_model.calibration)
        self.control_size = len(state_model.control)

        self.arglist_state = sorted(list(state_model.state), key=lambda x: x.name)
        self.arglist_calibration = sorted(
            list(state_model.calibration), key=lambda x: x.name
        )
        self.arglist_control = sorted(list(state_model.control), key=lambda x: x.name)
        self.arglist = (
            [state_model.dt]
            + self.arglist_state
            + self.arglist_calibration
            + self.arglist_control
        )

        self.State = named_vector("State", self.arglist_state)
        self.Covariance = named_covariance("Covariance", self.arglist_state)
        self.Control = named_vector("Control", self.arglist_control)
        self.Calibration = named_vector("Calibration", self.arglist_calibration)

        if self.calibration_size > 0:
            if len(calibration_map) == 0:
                map_lite = "\n  , ".join(
                    [f"{k}: ..." for k in self.arglist_calibration[:3]]
                )
                if len(self.arglist_calibration) > 3:
                    map_lite += ", ..."
                raise ModelConstructionError(
                    f"Model Missing specification of calibration_map:\n{{{map_lite}}}"
                )
            if len(calibration_map) != self.calibration_size:
                missing_from_map = set(state_model.calibration) - set(
                    calibration_map.keys()
                )
                extra_from_map = set(calibration_map.keys()) - set(
                    state_model.calibration
                )
                missing = ""
                if len(missing_from_map) > 0:
                    missing = f"\nMissing: {missing_from_map}"
                extra = ""
                if len(extra_from_map) > 0:
                    extra = f"\nExtra: {extra_from_map}"
                raise ModelConstructionError(f"Mismatched Calibration:{missing}{extra}")

        self._process_model = BasicBlock(
            statements=self._translate_process_model(state_model),
            indent=4,
            config=config,
        )
        self._process_jacobian = BasicBlock(
            statements=self._translate_process_jacobian(state_model),
            indent=4,
            config=config,
        )

        self._control_jacobian = BasicBlock(
            statements=self._translate_control_jacobian(state_model),
            indent=4,
            config=config,
        )
        self._control_covariance = BasicBlock(
            statements=self._translate_control_covariance(process_noise),
            indent=4,
            config=config,
        )

        # TODO(buck): Translate the sensor models dictionary contents into BasicBlocks
        self.sensorlist = sorted(
            [(k, v, sensor_noises[k]) for k, v in sensor_models.items()]
        )

        self._return = self._translate_return()

    def _translate_process_model(self, symbolic_model):
        subs_set = (
            [
                (
                    member,
                    Symbol("state.state.{}()".format(member)),
                )
                for member in self.arglist_state
            ]
            + [
                (
                    member,
                    Symbol("calibration.{}()".format(member)),
                )
                for member in self.arglist_calibration
            ]
            + [
                (
                    member,
                    Symbol("control.{}()".format(member)),
                )
                for member in self.arglist_control
            ]
        )

        for a in self.arglist_state:
            expr_before = symbolic_model.state_model[a]
            expr_after = expr_before.subs(subs_set)
            yield f"double {a.name}", expr_after

    def process_model_body(self):
        yield from self._process_model.compile()
        yield Return(self._return)

    def _translate_process_jacobian(self, symbolic_model):
        subs_set = (
            [
                (
                    member,
                    Symbol("state.state.{}()".format(member)),
                )
                for member in self.arglist_state
            ]
            + [
                (
                    member,
                    Symbol("calibration.{}()".format(member)),
                )
                for member in self.arglist_calibration
            ]
            + [
                (
                    member,
                    Symbol("control.{}()".format(member)),
                )
                for member in self.arglist_control
            ]
        )

        for idx, symbol in enumerate(self.arglist_state):
            model = symbolic_model.state_model[symbol]
            for state_idx, state in enumerate(self.arglist_state):
                assignment = f"jacobian({idx}, {state_idx})"
                expr_before = diff(model, state)
                expr_after = expr_before.subs(subs_set)
                yield assignment, expr_after

    def process_jacobian_body(self):
        yield MemberDeclaration("ExtendedKalmanFilter::ProcessJacobianT", "jacobian")
        yield from self._process_jacobian.compile()
        yield Return("jacobian")

    def _translate_control_jacobian(self, symbolic_model):
        subs_set = (
            [
                (
                    member,
                    Symbol("state.state.{}()".format(member)),
                )
                for member in self.arglist_state
            ]
            + [
                (
                    member,
                    Symbol("calibration.{}()".format(member)),
                )
                for member in self.arglist_calibration
            ]
            + [
                (
                    member,
                    Symbol("control.{}()".format(member)),
                )
                for member in self.arglist_control
            ]
        )

        for idx, symbol in enumerate(self.arglist_state):
            model = symbolic_model.state_model[symbol]
            for control_idx, control in enumerate(self.arglist_control):
                assignment = f"jacobian({idx}, {control_idx})"
                expr_before = diff(model, control)
                expr_after = expr_before.subs(subs_set)
                yield assignment, expr_after

    def control_jacobian_body(self):
        yield MemberDeclaration("ExtendedKalmanFilter::ControlJacobianT", "jacobian")
        yield from self._control_jacobian.compile()
        yield Return("jacobian")

    def _translate_return(self):
        content = ", ".join(
            ".{name}={name}".format(name=name) for name in self.arglist_state
        )
        return "State({" + content + "});"

    def enable_control(self):
        return self.control_size > 0

    def _translate_control_covariance(self, covariance):
        for i, iKey in enumerate(self.arglist_control):
            for j, jKey in enumerate(self.arglist_control):
                if (iKey, jKey) in covariance:
                    value = covariance[(iKey, jKey)]
                elif (jKey, iKey) in covariance:
                    value = covariance[(jKey, iKey)]
                elif i == j and iKey in covariance:
                    value = covariance[iKey]
                else:
                    value = 0.0

                yield f"covariance({i}, {j})", value
                if i != j:
                    yield f"covariance({j}, {i})", value

    def control_covariance_body(self):
        yield MemberDeclaration("ExtendedKalmanFilter::CovarianceT", "covariance")
        yield from self._control_covariance.compile()
        yield Return("covariance")

    def enable_calibration(self):
        return self.calibration_size > 0

    def _translate_sensor_model(self, sensor_model_mapping):
        subs_set = [
            (
                member,
                Symbol("state.state.{}()".format(member)),
            )
            for member in self.arglist_state
        ] + [
            (
                member,
                Symbol("calibration.{}()".format(member)),
            )
            for member in self.arglist_calibration
        ]
        for predicted_reading, model in sorted(list(sensor_model_mapping.items())):
            expr_before = model
            expr_after = expr_before.subs(subs_set)
            yield f"double {predicted_reading}", expr_after

    def reading_types(self, verbose=False):
        for name, sensor_model_mapping, sensor_noise in self.sensorlist:
            typename = name.title()
            identifier = f"SensorId::{name.upper()}"
            arglist_sensor = sorted(list(sensor_model_mapping.keys()))
            members = "\n".join(
                "double& %s() { return data(%d, 0); }" % (name, idx)
                for idx, name in enumerate(arglist_sensor)
            )
            size = len(sensor_model_mapping)
            if verbose:
                print(
                    f"reading_types: name: {name} reading_type: {typename} {identifier} members:\n{members}"
                )
                print("Model:")
                for predicted_reading, model in sorted(
                    list(sensor_model_mapping.items())
                ):
                    print(f"Modeling {predicted_reading} as function of state: {model}")

            body = BasicBlock(
                statements=self._translate_sensor_model(sensor_model_mapping),
                indent=4,
                config=self.config,
            )
            return_ = Return(
                "{}Options{{".format(typename)
                + ", ".join(
                    str(reading)
                    for reading in sorted(list(sensor_model_mapping.keys()))
                )
                + "}"
            )
            SensorModel_model_body = list(body.compile()) + [return_]

            SensorCovariance = named_covariance(f"{name}Covariance", arglist_sensor)

            SensorModel_covariance_body = self._translate_sensor_covariance(
                typename, SensorCovariance.from_dict(sensor_noise)
            )
            SensorModel_jacobian_body = self._translate_sensor_jacobian(
                typename, sensor_model_mapping
            )

            initializer_list = (
                "data("
                + ", ".join(
                    f"options.{name}"
                    for name in sorted(list(sensor_model_mapping.keys()))
                )
                + ")"
            )
            Options_members = "\n".join(
                f"double {str(symbol)} = 0.0;"
                for symbol in sorted(list(sensor_model_mapping.keys()))
            )

            yield ReadingT(
                identifier=identifier,
                initializer_list=initializer_list,
                members=members,
                Options_members=Options_members,
                SensorModel_covariance_body=SensorModel_covariance_body,
                SensorModel_jacobian_body=SensorModel_jacobian_body,
                SensorModel_model_body=SensorModel_model_body,
                size=size,
                typename=typename,
                sensor_model_mapping=sensor_model_mapping,
            )

    def _translate_sensor_jacobian_impl(self, sensor_model_mapping):
        subs_set = [
            (
                member,
                Symbol("state.state.{}()".format(member)),
            )
            for member in self.arglist_state
        ] + [
            (
                member,
                Symbol("calibration.{}()".format(member)),
            )
            for member in self.arglist_calibration
        ]

        for reading_idx, (_predicted_reading, model) in enumerate(
            sorted(list(sensor_model_mapping.items()))
        ):
            for state_idx, state in enumerate(self.arglist_state):
                assignment = f"jacobian({reading_idx}, {state_idx})"
                expr_before = diff(model, state)
                expr_after = expr_before.subs(subs_set)
                yield assignment, expr_after

    def _translate_sensor_jacobian(self, typename, sensor_model_mapping):
        yield MemberDeclaration(f"{typename}::SensorJacobianT", "jacobian")
        yield from BasicBlock(
            statements=self._translate_sensor_jacobian_impl(sensor_model_mapping),
            indent=4,
            config=self.config,
        ).compile()
        yield Return("jacobian")

    def _translate_sensor_covariance_impl(self, covariance):
        rows, cols = covariance.shape
        for i in range(rows):
            for j in range(cols):
                yield f"covariance({i}, {j})", covariance.data[i, j]

    def _translate_sensor_covariance(self, typename, covariance):
        yield MemberDeclaration(f"{typename}::CovarianceT", "covariance")
        yield from BasicBlock(
            statements=self._translate_sensor_covariance_impl(covariance),
            indent=4,
            config=self.config,
        ).compile()
        yield Return("covariance")
