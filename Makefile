ci:
	bash actions/ci.bash

test:
	bash actions/test.bash

feature_test:
	bash actions/feature_test.bash

format:
	bash actions/format.bash

PYTARGETS := $(shell find py/ featuretests/ experimental/ -type f -name '*.py')

lint:
	bash actions/format.bash
	# general code checks
	flake8 --version
	flake8 --config=common/setup.cfg py/ featuretests/ | grep -v "local variable '_"
	# security oriented checks
	bandit -c common/bandit.yaml -r py/ featuretests/
	# remove unused
	autoflake -i -r py/ featuretests/ experimental/
	# move to modern patterns
	pyupgrade $(PYTARGETS)
	# format docstrings
	pydocstringformatter -w py/formak/ py/test/formak/ py/test/unit/cpp/ py/test/unit/python/ experimental/ featuretests/
	# check writing rules
	proselint --config=common/proselint.json  docs/designs/*.md docs/formak/*.md docs/*.md
	# pre-commit
	pre-commit --version
	pre-commit run --all-files
	# interrogate
	interrogate --version
	interrogate -vv py/formak/


tidy:
	bash actions/tidy.bash

pipsetup:
	bash actions/pip_setup.bash
