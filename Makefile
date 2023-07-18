ci:
	bash actions/ci.bash

test:
	bash actions/test.bash

feature_test:
	bash actions/feature_test.bash

format:
	bash actions/format.bash

lint:
	bash actions/format.bash
	# general code checks
	flake8 --version
	flake8 --config=common/setup.cfg py/ featuretests/ | grep -v "local variable '_"
	# security oriented checks
	bandit -c common/bandit.yaml -r py/
	# remove unused
	autoflake -i -r py/ featuretests/
	# move to modern patterns
	pyupgrade $(ag --python -g "." py/ experimental/ featuretests/)
	# format docstrings
	pydocstringformatter -w $(ag --python -g "." py/ featuretests/)
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
