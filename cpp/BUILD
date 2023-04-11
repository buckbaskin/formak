cc_library(
    name = "utils-stats",
    hdrs = [
        "include/formak/utils/stats.h",
    ],
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
    deps = ["@eigen"],
)

cc_test(
    name = "utils-stats-test",
    srcs = ["test/utils/stats_test.cpp"],
    deps = [
        ":utils-stats",
        "@gtest//:gtest_main",
    ],
)
