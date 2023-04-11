# NEW
load("@bazel_tools//tools/build_defs/cc:action_names.bzl", "ACTION_NAMES")

# NEW
load(
    "@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl",
    "feature",
    "flag_group",
    "flag_set",
    "tool_path",
)

all_link_actions = [
    # NEW
    ACTION_NAMES.cpp_link_executable,
    ACTION_NAMES.cpp_link_dynamic_library,
    ACTION_NAMES.cpp_link_nodeps_dynamic_library,
]

def _impl(ctx):
    tool_paths = [
        tool_path(
            name = "gcc",
            path = "/usr/bin/clang-12",
        ),
        tool_path(
            name = "ld",
            path = "/usr/lib/llvm-12/bin/lld",
        ),
        tool_path(
            name = "ar",
            path = "/usr/bin/ar",
        ),
        tool_path(
            name = "cpp",
            path = "/usr/bin/clang++-12",
        ),
        tool_path(
            name = "gcov",
            path = "/bin/falseG",
        ),
        tool_path(
            name = "nm",
            path = "/bin/falseN",
        ),
        tool_path(
            name = "objdump",
            path = "/bin/falseO",
        ),
        tool_path(
            name = "strip",
            path = "/bin/falseS",
        ),
    ]

    features = [
        # NEW
        feature(
            name = "default_linker_flags",
            enabled = True,
            flag_sets = [
                flag_set(
                    actions = all_link_actions,
                    flag_groups = ([
                        flag_group(
                            flags = [
                                "-lstdc++",
                            ],
                        ),
                    ]),
                ),
            ],
        ),
    ]

    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        features = features,
        cxx_builtin_include_directories = [
            "/usr/lib/llvm-12/lib/clang/12.0.0/include",
            "/usr/lib/llvm-12/lib/clang/12.0.0/share",
            "/usr/lib/llvm-12/lib64/clang/12.0.0/include",
            "/usr/lib/llvm-12/lib/clang/12.0.1/include",
            "/usr/lib/llvm-12/lib/clang/12.0.1/share",
            "/usr/lib/llvm-12/lib64/clang/12.0.1/include",
            "/usr/lib/llvm-12/include/c++/v1",
            "/usr/include",
        ],
        toolchain_identifier = "local",
        host_system_name = "local",
        target_system_name = "local",
        target_cpu = "k8",
        target_libc = "unknown",
        compiler = "clang",
        abi_version = "unknown",
        abi_libc_version = "unknown",
        tool_paths = tool_paths,
    )

cc_toolchain_config = rule(
    implementation = _impl,
    attrs = {},
    provides = [CcToolchainConfigInfo],
)
