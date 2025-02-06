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
