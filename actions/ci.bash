bt() {
    bazel test --verbose_failures "$@" //formak/...
    RESULT=$?
    if [ $RESULT -ne 0 ] ; then
        echo " "
        echo "    BAZEL test failed $RESULT with args " "$@"
        echo " "
        exit 1
    else
        echo "    BAZEL test passed $RESULT with args " "$@"
    fi
}

bt
bt --config=clang
bt --config=clang --config=asan
# Need to work on ignoring GTest
# bt --config=clang --config=msan
bt --config=clang --config=tsan
bt --config=clang --config=ubsan
