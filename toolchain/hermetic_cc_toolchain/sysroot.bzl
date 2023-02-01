# Derived from bazelembedded/rules_cc_toolchain
# https://github.com/bazelembedded/rules_cc_toolchain/blob/8f9de1b0ea47876e3de6b4fc9d9660331139aaa1/LICENSE

def _no_op(ctx):
    pass

sysroot_package = rule(
    _no_op,
    doc = "Marks a package as a sysroot. This rule serves as a placeholder for\
other labels to point to.",
)
