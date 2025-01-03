make format ;

flake8 --config=common/setup.cfg  \
backend_cpp/ \
cpp/ \
featuretests/ \
languagesupport/ \
problemdefinition/ \
py/ \
runtime_py/ \
| grep -v "ui.py.*unused" | grep "unused";

bazel test  \
//backend_cpp/... \
//cpp/... \
//featuretests/... \
//languagesupport/... \
//problemdefinition/... \
//py/... \
//runtime_py/... \
--test_keep_going
