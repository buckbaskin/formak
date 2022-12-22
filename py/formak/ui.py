from sympy import Symbol, simplify, symbols


class Model:
    def __init__(
        self,
        dt,
        state,
        control,
        state_model,
        process_noise=None,
        sensor_models=None,
        sensor_noises=None,
        compile=False,
        *,
        debug_print=False
    ):
        self.dt = dt
        self.state = state
        self.control = control
        self.state_model = state_model

        for k in list(self.state_model.keys()):
            self.state_model[k] = simplify(self.state_model[k])

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
