from backend_cpp.ast import Return
from backend_cpp.basic_block import BasicBlock
from backend_cpp.config import Config
from backend_py.named_vector import named_vector
from formak.exceptions import ModelConstructionError
from sympy import Symbol


class Model:
    """C++ implementation of the model."""

    def __init__(
        self, symbolic_model, calibration_map, namespace, header_include, config
    ):
        # TODO(buck): Enable mypy for type checking
        # TODO(buck): Move all type assertions to either __init__ (constructor) or mypy?
        # assert isinstance(symbolic_model, UiModel)
        if isinstance(config, dict):
            config = Config(**config)
        assert isinstance(config, Config)

        self.enable_EKF = False
        self.config = config
        self.namespace = namespace
        self.header_include = header_include

        self.sensorlist = {}

        self.state_size = len(symbolic_model.state)
        self.control_size = len(symbolic_model.control)
        self.calibration_size = len(symbolic_model.calibration)

        self.arglist_state = sorted(list(symbolic_model.state), key=lambda x: x.name)
        self.arglist_calibration = sorted(
            list(symbolic_model.calibration), key=lambda x: x.name
        )
        self.arglist_control = sorted(
            list(symbolic_model.control), key=lambda x: x.name
        )
        self.arglist = (
            [symbolic_model.dt]
            + self.arglist_state
            + self.arglist_calibration
            + self.arglist_control
        )

        self.State = named_vector("State", self.arglist_state)
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
                    f"Model with empty specification of calibration_map:\n{{{map_lite}}}"
                )
            if len(calibration_map) != self.calibration_size:
                missing_from_map = set(symbolic_model.calibration) - set(
                    calibration_map.keys()
                )
                extra_from_map = set(calibration_map.keys()) - set(
                    symbolic_model.calibration
                )
                missing = ""
                if len(missing_from_map) > 0:
                    missing = f"\nMissing: {missing_from_map}"
                extra = ""
                if len(extra_from_map) > 0:
                    extra = f"\nExtra: {extra_from_map}"
                raise ModelConstructionError(f"Mismatched Calibration:{missing}{extra}")

        self._model = BasicBlock(
            statements=self._translate_model(symbolic_model), indent=4, config=config
        )

        self._return = self._translate_return()

    def _translate_model(self, symbolic_model):
        subs_set = (
            [
                (
                    member,
                    Symbol("state.{}()".format(member)),
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

    def _translate_return(self):
        content = ", ".join(str(symbol) for symbol in self.arglist_state)
        return "State({" + content + "})"

    def model_body(self):
        yield from self._model.compile()
        yield Return(self._return)

    def enable_control(self):
        return self.control_size > 0

    def enable_calibration(self):
        return self.calibration_size > 0
