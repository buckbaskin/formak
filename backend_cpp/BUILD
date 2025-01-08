package(default_visibility = ["//visibility:private"])

load("@pip_deps//:requirements.bzl", "requirement")
load("//py:defs.bzl", "CC_TEST_DEPS", "PY_TEST_DEPS", "cc_formak_model", "py_test_suite")

filegroup(
    name = "templates",
    srcs = glob(
        [
            "formak/templates/*.hpp",
            "formak/templates/*.cpp",
            "formak/templates/test/*.cpp",
        ],
        allow_empty = False,
    ),
    visibility = ["//visibility:public"],
)

py_library(
    name = "ast",
    srcs = ["backend_cpp/ast.py"],
    imports = ["."],
    visibility = [
        "//featuretests/ast_code_generation:__pkg__",
    ],
    deps = [
        requirement("Jinja2"),
    ],
)

py_test_suite(
    name = "ast-test",
    srcs = [
        "test/ast_test.py",
    ],
    deps = [
        ":ast",
    ] + PY_TEST_DEPS,
)

py_library(
    name = "ast-fragments",
    srcs = ["backend_cpp/ast_fragments.py"],
    imports = ["."],
    deps = [
        ":ast",
    ],
)

py_library(
    name = "config",
    srcs = ["backend_cpp/config.py"],
    imports = ["."],
    visibility = [
        "//py:__pkg__",
    ],
)

py_library(
    name = "basic-block",
    srcs = ["backend_cpp/basic_block.py"],
    imports = ["."],
    visibility = [
        "//py:__pkg__",
    ],
)

py_library(
    name = "model",
    srcs = ["backend_cpp/model.py"],
    imports = ["."],
    visibility = [
        "//py:__pkg__",
    ],
    deps = [":basic-block"],
)

py_library(
    name = "extended-kalman-filter",
    srcs = ["backend_cpp/extended_kalman_filter.py"],
    imports = ["."],
    visibility = [
        "//py:__pkg__",
    ],
    deps = [":basic-block"],
)

py_library(
    name = "compile-result",
    srcs = ["backend_cpp/compile_result.py"],
    imports = ["."],
    visibility = [
        "//py:__pkg__",
    ],
)

# TODO split this
py_library(
    name = "cpp",
    srcs = ["backend_cpp/cpp.py"],
    imports = ["."],
    visibility = [
        "//py:__pkg__",
    ],
    deps = [
        ":ast",
        ":ast-fragments",
        ":basic-block",
        ":compile-result",
        ":config",
        ":extended-kalman-filter",
        ":model",
        "//backend_py:named-covariance",
        "//backend_py:named-vector",
        "//frontend:model-validation",
        "//py:exceptions",
        requirement("sympy"),
    ],
)

py_test_suite(
    name = "compile-ekf-test",
    srcs = [
        "test/unit/cpp/compile_ekf_test.py",
    ],
    deps = [
        ":cpp",
        ":config",
        "//py:exceptions",
        "//problemdefinition:model-for-testing",
        requirement("sympy"),
    ] + PY_TEST_DEPS,
)

cc_formak_model(
    name = "simple-model",
    namespace = "unit",
    pydeps = [],
    pymain = "test/unit/cpp/generator.py",
    pysrcs = glob(
        ["test/unit/cpp/*.py"],
        allow_empty = False,
    ),
)

cc_formak_model(
    name = "simple-ekf",
    namespace = "unit",
    pydeps = [],
    pymain = "test/unit/cpp/generator_ekf.py",
    pysrcs = glob(
        ["test/unit/cpp/*.py"],
        allow_empty = False,
    ),
)

cc_formak_model(
    name = "ekf-cse",
    namespace = "cse",
    pydeps = [],
    pymain = "test/unit/cpp/generator_ekf_cse.py",
    pysrcs = glob(
        ["test/unit/cpp/*.py"],
        allow_empty = False,
    ),
)

cc_formak_model(
    name = "ekf-no-cse",
    namespace = "no_cse",
    pydeps = [],
    pymain = "test/unit/cpp/generator_ekf_no_cse.py",
    pysrcs = glob(
        ["test/unit/cpp/*.py"],
        allow_empty = False,
    ),
)

cc_formak_model(
    name = "calibration-model",
    namespace = "unit",
    pydeps = [],
    pymain = "test/unit/cpp/generator_with_calibration.py",
    pysrcs = glob(
        ["test/unit/cpp/*.py"],
        allow_empty = False,
    ),
)

cc_formak_model(
    name = "calibration-ekf",
    namespace = "unit",
    pydeps = [],
    pymain = "test/unit/cpp/generator_ekf_with_calibration.py",
    pysrcs = glob(
        ["test/unit/cpp/*.py"],
        allow_empty = False,
    ),
)

cc_test(
    name = "unit-cpp-extended-kalman-filter-test",
    srcs = ["test/unit/cpp/ExtendedKalmanFilter_test.cpp"],
    deps = [
        ":simple-ekf",
    ] + CC_TEST_DEPS,
)

cc_test(
    name = "unit-cpp-basic-block",
    srcs = ["test/unit/cpp/BasicBlock_test.cpp"],
    deps = [
        ":ekf-cse",
        ":ekf-no-cse",
    ] + CC_TEST_DEPS,
)

cc_test(
    name = "unit-cpp-model-test",
    srcs = [
        "test/unit/cpp/Model_test.cpp",
    ],
    deps = [
        ":simple-model",
    ] + CC_TEST_DEPS,
)

cc_test(
    name = "unit-cpp-calibration-model-test",
    srcs = [
        "test/unit/cpp/ModelWithCalibration_test.cpp",
    ],
    deps = [
        ":calibration-model",
    ] + CC_TEST_DEPS,
)

cc_test(
    name = "unit-cpp-calibration-ekf-test",
    srcs = [
        "test/unit/cpp/ExtendedKalmanFilterWithCalibration_test.cpp",
    ],
    deps = [
        ":calibration-ekf",
    ] + CC_TEST_DEPS,
)

cc_test(
    name = "integration-cpp-model-test",
    srcs = ["test/integration/Model_test.cpp"],
    deps = [
        ":simple-model",
        "//cpp:utils-stats",
        "@rapidcheck",
    ] + CC_TEST_DEPS,
)

cc_test(
    name = "integration-cpp-extended-kalman-filter-test",
    srcs = ["test/integration/ExtendedKalmanFilter_test.cpp"],
    deps = [
        ":simple-ekf",
        "//cpp:utils-stats",
        "@rapidcheck",
    ] + CC_TEST_DEPS,
)
