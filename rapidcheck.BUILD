cc_library(
    name = "include-impl",
    srcs = glob(
        [
            "src/BeforeMinimalTestCase.cpp",
            "src/Show.cpp",
            "src/gen/detail/*.cpp",
        ],
    ),
    hdrs = glob([
        "include/rapidcheck/**/*.h",
        "include/rapidcheck/**/*.hpp",
        "include/rapidcheck/*.h",
        "include/rapidcheck/*.hpp",
    ]),
    strip_include_prefix = "include",
    visibility = ["//visibility:private"],
)

cc_library(
    name = "src-detail",
    srcs = glob(
        ["src/detail/*.cpp"],
    ),
    hdrs = glob([
        "src/detail/*.h",
    ]),
    strip_include_prefix = "src",
    visibility = ["//visibility:private"],
    deps = [":include-impl"],
)

cc_library(
    name = "rapidcheck",
    srcs = glob(
        [
            "src/*.cpp",
            "src/gen/*.cpp",
        ],
    ),
    hdrs = glob([
        "include/*.h",
    ]),
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
    deps = [":src-detail"],
)
