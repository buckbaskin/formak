if [ $# -eq 0 ]; then
    DEFAULT=$(pwd)
else
    DEFAULT=$1
fi

echo Formatting Directory $DEFAULT

bazel run --config=clang //tools:buildifier -- -r $DEFAULT ;
black $DEFAULT ;
isort --profile black py/ ;
SEARCHRESULT=$(ag --cpp -g ".*" $DEFAULT) ;
clang-format-12 -i -style=file $SEARCHRESULT
