cc_library(
    name = "innovation-filtering",
    hdrs = [
        "include/formak/innovation_filtering.h",
    ],
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
    deps = ["@eigen"],
)

cc_test(
    name = "innovation-filtering-test",
    srcs = ["test/innovation_filtering_test.cpp"],
    deps = [
        ":innovation-filtering",
        "@gtest//:gtest_main",
    ],
)
