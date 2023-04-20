from datetime import datetime

from formak.exceptions import ModelDefinitionError
from sympy import Matrix, Symbol, simplify, symbols


class Model:
    def __init__(
        self,
        dt,
        state,
        calibration,
        control,
        state_model,
        compile=False,
        *,
        proactive_simplify=False,
        debug_print=False,
    ):
        start_time = datetime.now()

        self.dt = dt
        self.state = state
        self.calibration = calibration
        self.control = control
        self.state_model = state_model

        if not set(self.state).isdisjoint(set(self.calibration)):
            raise ModelDefinitionError(
                f"States shared between state, calibration: {set(self.state).intersection(set(self.calibration))}"
            )

        if not set(self.state).isdisjoint(set(self.control)):
            raise ModelDefinitionError(
                f"States shared between state, control: {set(self.state).intersection(set(self.control))}"
            )

        if not set(self.calibration).isdisjoint(set(self.control)):
            raise ModelDefinitionError(
                f"States shared between calibration, control: {set(self.calibration).intersection(set(self.control))}"
            )

        if not len(state_model) == len(state):
            raise ModelDefinitionError(
                f"State Model of size {len(state_model)} does not match State of size {len(state)}"
            )

        for k in state:
            try:
                assert k in state_model
            except AssertionError:
                print("{} ( {} ) missing from state model".format(k, type(k)))
                raise

        if proactive_simplify:
            print(f"pre simplify {datetime.now() - start_time}")
            for idx, k in enumerate(
                sorted(list(self.state_model.keys()), key=lambda x: x.name)
            ):
                pre = self.state_model[k]
                self.state_model[k] = simplify(self.state_model[k])
                print(f"{idx} {k} {datetime.now() - start_time}")

            print(f"post simplify {datetime.now() - start_time}")

        if debug_print:
            print("State Model")
            for k in sorted(list(state_model.keys()), key=lambda x: x.name):
                print("  {}: {}".format(k, state_model[k]))
