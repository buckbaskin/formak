bash actions/format.bash

flake8 --config=common/setup.cfg py/ | grep -v "local variable '_"

bandit -c common/bandit.yaml -r py/
