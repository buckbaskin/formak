class FormakBaseException(Exception):
    pass


class MinimizationFailure(FormakBaseException):
    def __init__(self, minimization_result, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.success = minimization_result.success
        self.message = minimization_result.message

    def __str__(self):
        return "MinimizationFailure({}, {})".format(
            self.success,
            self.message,
        )

    def __repr__(self):
        return "MinimizationFailure({}, {})".format(
            self.success,
            self.message,
        )


# UI Model
class ModelDefinitionError(FormakBaseException):
    pass


# Python / C++ Compiled Model
class ModelConstructionError(FormakBaseException):
    pass
