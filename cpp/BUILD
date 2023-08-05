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
    name = "utils-type-checks",
    hdrs = [
        "include/formak/utils/type_checks.h",
    ],
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
    deps = ["@eigen"],
)

cc_test(
    name = "utils-type-checks-test",
    srcs = ["test/utils/type_checks_test.cpp"],
    deps = [
        ":utils-type-checks",
        "@gtest//:gtest_main",
    ],
)
