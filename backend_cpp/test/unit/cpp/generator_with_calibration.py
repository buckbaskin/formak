from backend_cpp import cpp

from formak.problemdefinition.model import Model as UiModel
from sympy import Symbol, symbols

dt, a, b, x = symbols(["dt", "a", "b", "x"])

ui_model = UiModel(
    dt=dt,
    state={x},
    control=set(),
    calibration={a, b},
    state_model={x: x + a + b},
)

cpp_implementation = cpp.compile(
    ui_model, calibration_map={Symbol("a"): 5.0, Symbol("b"): 0.5}, config={}
)

print("Wrote header at path {}".format(cpp_implementation.header_path))
print("Wrote source at path {}".format(cpp_implementation.source_path))
