class Model(object):
    """Python implementation of the model"""

    def __init__(self, symbolic_model, config):
        # assert isinstance(symbolic_model, ui.Model)

        self.arglist = sorted(
            list(symbolic_model.state), key=lambda x: x.name
        ) + sorted(list(symbolic_model.control), key=lambda x: x.name)


class ExtendedKalmanFilter(object):
    def __init__(self, state_model, sensor_models):
        self.state_model = state_model
        self.sensor_models = sensor_models


class Config(object):
    def __init__(self, compile):
        self.compile = compile


def default_config():
    return Config(compile=False)


def compile(symbolic_model, config=None):
    if config is None:
        config = default_config()
    elif isinstance(config, dict):
        config = Config(**config)

    return Model(symbolic_model, config)
