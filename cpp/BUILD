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

cc_library(
    name = "utils-microbenchmark",
    srcs = ["src/utils/microbenchmark.cpp"],
    hdrs = [
        "include/formak/utils/microbenchmark.h",
    ],
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
)

cc_test(
    name = "utils-microbenchmark-test",
    srcs = ["test/utils/microbenchmark_test.cpp"],
    deps = [
        ":utils-microbenchmark",
        "@gtest//:gtest_main",
    ],
)

cc_library(
    name = "innovation-filtering",
    hdrs = [
        "include/formak/innovation_filtering.h",
    ],
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
    deps = [],
)
