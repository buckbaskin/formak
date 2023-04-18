from datetime import datetime

from sympy import Matrix, Symbol, simplify, symbols


class Model:
    def __init__(
        self, dt, state, control, state_model, compile=False, *, debug_print=False
    ):
        start_time = datetime.now()

        self.dt = dt
        self.state = state
        self.control = control
        self.state_model = state_model

        print(f"pre simplify {datetime.now() - start_time}")
        for idx, k in enumerate(sorted(list(self.state_model.keys()))):
            self.state_model[k] = simplify(self.state_model[k])
            print(f"{idx} {k} {datetime.now() - start_time}")
            if idx >= 5:
                1 / 0

        print(f"post simplify {datetime.now() - start_time}")
        1 / 0

        assert len(state_model) == len(state)

        for k in state:
            try:
                assert k in state_model
            except AssertionError:
                print("{} ( {} ) missing from state model".format(k, type(k)))
                raise

        if debug_print:
            print("State Model")
            for k in sorted(list(state_model.keys()), key=lambda x: x.name):
                print("  {}: {}".format(k, state_model[k]))
