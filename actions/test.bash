bt() {
    bazel test --verbose_failures "$@" //py/... //cpp/... //languagesupport/...
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

bt --config=clang
