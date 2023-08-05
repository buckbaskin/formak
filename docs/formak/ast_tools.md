# AST Tools

# For Users

Start with either a `HeaderFile` or `SourceFile`, along with at least one
`namespace`.

Then keep building up the tree structure recursively. I recommend starting with
`ast_fragments` and only implementing new combinations when necessary.

If you don't think something is supported, I'd recommend starting with
`FromFileTemplate`. If you need to insert an arbitrary string, you can pass it
wrapped via `Escape`.

# For Contributors

Every class should inherit from BaseAst

Future Extensions:
- Visitor, Editor patterns for viewing or building the tree
- Split out statements and expressions
- Split out constructs that can only be used in a header file, source file or both
- Split out constructs based on what scopes they can be used
