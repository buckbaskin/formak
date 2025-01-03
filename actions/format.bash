#!/usr/bin/env bash

if [ $# -eq 0 ]; then
    DEFAULT=$(pwd)
else
    DEFAULT=$1
fi

echo Formatting Directory $DEFAULT

echo "buildifier"
bazel run --config=clang //tools:buildifier -- -r $DEFAULT ;

echo "black"
black $DEFAULT ;

echo "isort"
isort --settings-path=common/ --profile black \
backend_cpp/ \
backend_py/ \
cpp/ \
featuretests/ \
frontend/ \
languagesupport/ \
problemdefinition/ \
py/ \
runtime_py/ \
; echo "done isort"

# echo "codespell"
# codespell py/ featuretests/ languagesupport/

echo "clang-format"
SEARCHRESULT=$(ag --cpp -g ".*" $DEFAULT) ;
clang-format-12 -i -style=file $SEARCHRESULT

echo "pre-commit"
pre-commit install
