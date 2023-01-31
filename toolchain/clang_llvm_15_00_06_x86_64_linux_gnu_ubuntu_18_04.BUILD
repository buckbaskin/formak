# Derived from bazelembedded/rules_cc_toolchain
# https://github.com/bazelembedded/rules_cc_toolchain/blob/8f9de1b0ea47876e3de6b4fc9d9660331139aaa1/LICENSE

load(
    "@formak//cc_toolchain:cc_toolchain_import.bzl",
    "cc_toolchain_import",
)

exports_files(glob(["bin/*"]))

filegroup(
    name = "all",
    srcs = glob(["**/*"]),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "ar_files",
    srcs = ["bin/llvm-ar"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "compiler_files",
    srcs = [
        "bin/clang",
        "bin/clang++",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "linker_files",
    srcs = [
        "bin/ld.lld",
    ],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "objcopy_files",
    srcs = ["bin/llvm-objcopy"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "strip_files",
    srcs = ["bin/llvm-strip"],
    visibility = ["//visibility:public"],
)

cc_toolchain_import(
    name = "llvm_libunwind",
    hdrs = ["lib/clang/12.0.0/include/unwind.h"],
    includes = ["lib/clang/12.0.0/include"],
    runtime_path = "/usr/lib/x86_64-linux-gnu",
    shared_library = "lib/libunwind.so",
    static_library = "lib/libunwind.a",
    target_compatible_with = select({
        "@platforms//os:linux": ["@platforms//cpu:x86_64"],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    visibility = ["@formak//config:__pkg__"],
    deps = [
        "@formak_config//:libc",
    ],
)

cc_toolchain_import(
    name = "llvm_libcxx",
    hdrs = glob(["include/c++/v1/**"]),
    includes = ["include/c++/v1"],
    static_library = "lib/libc++.a",
    target_compatible_with = select({
        "@platforms//os:linux": ["@platforms//cpu:x86_64"],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    visibility = ["@formak//config:__pkg__"],
    deps = [
        "@formak_config//:libc",
        "@formak_config//:libunwind",
    ],
)

cc_toolchain_import(
    name = "llvm_libcxx_abi",
    hdrs = glob(["include/c++/v1/**"]),
    includes = ["include/c++/v1"],
    static_library = "lib/libc++abi.a",
    target_compatible_with = select({
        "@platforms//os:linux": ["@platforms//cpu:x86_64"],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    visibility = ["@formak//config:__pkg__"],
    deps = [
        "@formak_config//:libc",
        "@formak_config//:libpthread",
    ],
)

cc_toolchain_import(
    name = "llvm_libclang_rt",
    hdrs = glob([
        "lib/clang/12.0.0/*.h",
        "lib/clang/12.0.0/include/*.h",
        "lib/clang/12.0.0/include/**/*.h",
    ]),
    includes = [
        "lib/clang/12.0.0",
        "lib/clang/12.0.0/include",
    ],
    static_library = "lib/clang/12.0.0/lib/linux/libclang_rt.builtins-x86_64.a",
    target_compatible_with = select({
        "@platforms//os:linux": ["@platforms//cpu:x86_64"],
        "//conditions:default": ["@platforms//:incompatible"],
    }),
    visibility = ["@formak//config:__pkg__"],
)
