from formak import cpp, ui

model = ui.Model(
    ui.Symbol("dt"),
    set(ui.symbols(["x", "y"])),
    set(ui.symbols(["a"])),
    {ui.Symbol("x"): "x * y", ui.Symbol("y"): "y + a * dt"},
)

cpp_implementation = cpp.compile(model)

print("Wrote header at path {}".format(cpp_implementation.header_path))
print("Wrote source at path {}".format(cpp_implementation.source_path))
