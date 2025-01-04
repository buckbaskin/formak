from backend_cpp import cpp
from backend_cpp.config import Config

from formak.problemdefinition.model import Model as UiModel
from sympy import Symbol, symbols

cpp_implementation = cpp.compile_ekf(
    state_model=UiModel(
        dt=Symbol("dt"),
        state=set(symbols(["x", "y"])),
        control=set(symbols(["a"])),
        # Add 1e-3 * a to prevent (0, 0) from having no variance
        state_model={
            Symbol("x"): "x + y * a * dt",
            Symbol("y"): "y + x * a * dt",
        },
    ),
    process_noise={Symbol("a"): 0.25},
    sensor_models={
        "simple": {"reading1": Symbol("x")},
        "combined": {"reading2": Symbol("x") + Symbol("y")},
    },
    sensor_noises={"simple": {"reading1": 1.0}, "combined": {"reading2": 4.0}},
    config=Config(common_subexpression_elimination=False),
)

print("Wrote header at path {}".format(cpp_implementation.header_path))
print("Wrote source at path {}".format(cpp_implementation.source_path))
