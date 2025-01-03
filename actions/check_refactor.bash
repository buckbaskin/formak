make format ;

echo ">>> flake8" ;
flake8 --config=common/setup.cfg  \
backend_cpp/ \
backend_py/ \
cpp/ \
featuretests/ \
frontend/ \
languagesupport/ \
problemdefinition/ \
py/ \
runtime_py/ \
| grep -v "ui.py.*unused" | grep "unused";
echo "<<< flake8" ;

echo ">>> test" ;
bazel test  \
//backend_cpp/... \
//backend_py/... \
//cpp/... \
//featuretests/... \
//frontend/... \
//languagesupport/... \
//problemdefinition/... \
//py/... \
//runtime_py/... \
--test_keep_going ;
echo "<<< test" ;
