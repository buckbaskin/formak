# Derived from bazelembedded/rules_cc_toolchain
# https://github.com/bazelembedded/rules_cc_toolchain/blob/8f9de1b0ea47876e3de6b4fc9d9660331139aaa1/LICENSE

def register_cc_toolchains():
    native.register_toolchains("@formak//toolchain/hermetic_cc_toolchain/...")
