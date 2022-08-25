cc_library(
    name = "main",
    srcs = glob(
        ["src/*.cc"],
        exclude = ["src/gtest-all.cc"],
    ),
    hdrs = glob([
        "include/**/*.h",
        "src/*.h",
    ]),
    copts = ["-Iexternal/gtest/include"],
    linkopts = [
        "-lm",
        "-pthread",
    ],
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)
