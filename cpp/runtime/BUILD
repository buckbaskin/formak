cc_library(
    name = "managed-filter",
    hdrs = [
        "include/formak/runtime/ManagedFilter.h",
    ],
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
    deps = [],
)

cc_test(
    name = "managed-filter-test",
    srcs = ["test/ManagedFilter_test.cpp"],
    deps = [
        ":managed-filter",
        "@gtest//:gtest_main",
    ],
)
