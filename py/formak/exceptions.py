class FormakBaseException(Exception):
    pass


class MinimizationFailure(FormakBaseException):
    def __init__(self, minimization_result, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.success = minimization_result.success
        self.message = minimization_result.message

    def __repr__(self):
        return "MinimizationFailure({}, {})".format(
            self.success,
            self.message,
        )
