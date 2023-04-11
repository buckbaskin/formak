bt() {
    bazel test --verbose_failures "$@" //featuretests/...
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
# bt --config=clang
