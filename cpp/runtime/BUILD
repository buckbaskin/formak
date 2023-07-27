cc_library(
    name = "managed-filter",
    hdrs = [
        "include/formak/runtime/ManagedFilter.h",
    ],
    strip_include_prefix = "include",
    visibility = ["//visibility:public"],
    deps = [],
)

cc_library(
    name = "test-tools",
    testonly = True,
    srcs = [
        "src/test/tools.cpp",
    ],
    hdrs = [
        "include/formak/runtime/test/tools.h",
    ],
    strip_include_prefix = "include",
    visibility = ["//visibility:private"],
    deps = [],
)

cc_test(
    name = "managed-filter-test",
    srcs = ["test/ManagedFilter_test.cpp"],
    deps = [
        ":managed-filter",
        ":test-tools",
        "@gtest//:gtest_main",
    ],
)

cc_test(
    name = "managed-filter-no-control-test",
    srcs = ["test/ManagedFilterNoControl_test.cpp"],
    deps = [
        ":managed-filter",
        ":test-tools",
        "@gtest//:gtest_main",
    ],
)
