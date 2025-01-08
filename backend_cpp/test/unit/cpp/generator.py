from backend_cpp.compile_model import compile_model
from formak.problemdefinition.model import Model as UiModel
from sympy import Symbol, symbols

model = UiModel(
    Symbol("dt"),
    set(symbols(["x", "y"])),
    set(symbols(["a"])),
    {Symbol("x"): "x * y", Symbol("y"): "y + a * dt"},
)

cpp_implementation = compile_model(model)

print("Wrote header at path {}".format(cpp_implementation.header_path))
print("Wrote source at path {}".format(cpp_implementation.source_path))
