# Derived from bazelembedded/rules_cc_toolchain
# https://github.com/bazelembedded/rules_cc_toolchain/blob/8f9de1b0ea47876e3de6b4fc9d9660331139aaa1/LICENSE

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@formak//config:rules_cc_toolchain_config_repository.bzl", "rules_cc_toolchain_config")

def rules_cc_toolchain_deps():
    """Fetches the toolchain dependencies """

    # Setup clang compiler files.
    # Required by: rules_cc_toolchain.
    # Used by modules: cc_toolchain.
    if "clang_llvm_15_00_06_x86_64_linux_gnu_ubuntu_18_04" not in native.existing_rules():
        http_archive(
            name = "clang_llvm_15_00_06_x86_64_linux_gnu_ubuntu_18_04",
            url = "https://github.com/llvm/llvm-project/releases/download/llvmorg-15.0.6/clang+llvm-15.0.6-x86_64-linux-gnu-ubuntu-18.04.tar.xz",
            sha256 = "1234",
            build_file = "//toolchain:clang_llvm_15_00_06_x86_64_linux_gnu_ubuntu_18_04.BUILD",
            strip_prefix = "clang+llvm-15.0.6-x86_64-linux-gnu-ubuntu-18.04",
        )

    # Setup os normalisation tools.
    # Required by: rules_cc_toolchain.
    # Required by modules: cc_toolchain/internal/include_tools.
    if "rules_os" not in native.existing_rules():
        git_repository(
            name = "rules_os",
            commit = "68cdf228f8449a2b42b3a7b6d65395af74a007d7",
            remote = "https://github.com/silvergasp/rules_os.git",
        )

    # Setup x64 linux sysroot
    # Required by: rules_cc_toolchain, rules_cc_toolchain_config.
    # Required by modules: cc_toolchain.
    if "debian_stretch_amd64_sysroot" not in native.existing_rules():
        http_archive(
            name = "debian_stretch_amd64_sysroot",
            sha256 = "84656a6df544ecef62169cfe3ab6e41bb4346a62d3ba2a045dc5a0a2ecea94a3",
            urls = ["https://commondatastorage.googleapis.com/chrome-linux-sysroot/toolchain/2202c161310ffde63729f29d27fe7bb24a0bc540/debian_stretch_amd64_sysroot.tar.xz"],
            build_file = "//toolchain:debian_stretch_amd64_sysroot.BUILD",
        )

    # Setup default configuration for toolchain.
    # Required by: rules_cc_toolchain.
    # Required by modules: third_party, cc_toolchain.
    if "rules_cc_toolchain_config" not in native.existing_rules():
        rules_cc_toolchain_config(
            name = "rules_cc_toolchain_config",
        )
