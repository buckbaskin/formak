# Derived from bazelembedded/rules_cc_toolchain
# https://github.com/bazelembedded/rules_cc_toolchain/blob/8f9de1b0ea47876e3de6b4fc9d9660331139aaa1/LICENSE

package(default_visibility = ["//visibility:public"])

# Compilers
label_flag(
    name = "clang",
    build_setting_default = "@clang_llvm_12_00_x86_64_linux_gnu_ubuntu_16_04//:all",
)

# Libraries
label_flag(
    name = "libc",
    build_setting_default = "@rules_cc_toolchain//config:libc_multiplexer",
)

label_flag(
    name = "libpthread",
    build_setting_default = "@rules_cc_toolchain//config:pthread_multiplexer",
)

label_flag(
    name = "libunwind",
    build_setting_default = "@rules_cc_toolchain//config:libunwind_multiplexer",
)

label_flag(
    name = "libc++",
    build_setting_default = "@rules_cc_toolchain//config:libc++_multiplexer",
)

label_flag(
    name = "libc++abi",
    build_setting_default = "@rules_cc_toolchain//config:libc++abi_multiplexer",
)

label_flag(
    name = "compiler_rt",
    build_setting_default = "@rules_cc_toolchain//config:libclang_rt_multiplexer",
)

label_flag(
    name = "user_defined",
    build_setting_default = "@rules_cc_toolchain//config:empty",
)

label_flag(
    name = "startup_libs",
    build_setting_default = "@rules_cc_toolchain//config:startup_libs",
)

label_flag(
    name = "clang_tidy_config",
    build_setting_default = "@rules_cc_toolchain//config:clang_tidy_config_multiplexer",
)
