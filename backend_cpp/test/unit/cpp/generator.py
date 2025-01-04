from backend_cpp import cpp

from formak.problemdefinition.model import Model as UiModel
from sympy import Symbol, symbols

model = UiModel(
    Symbol("dt"),
    set(symbols(["x", "y"])),
    set(symbols(["a"])),
    {Symbol("x"): "x * y", Symbol("y"): "y + a * dt"},
)

cpp_implementation = cpp.compile(model)

print("Wrote header at path {}".format(cpp_implementation.header_path))
print("Wrote source at path {}".format(cpp_implementation.source_path))
