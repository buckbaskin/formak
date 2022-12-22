bash actions/format.bash

echo "flake8"
# general code checks
flake8 --config=common/setup.cfg py/ | grep -v "local variable '_"

echo "bandit"
# security oriented checks
bandit -c common/bandit.yaml -r py/

echo "autoflake"
# remove unused
autoflake -i -r py/

echo "pyupgrade"
# move to modern patterns
pyupgrade $(ag --python -g "." py/)

echo "pydocstringformatter"
# format docstrings
pydocstringformatter -w $(ag --python -g "." py/)
