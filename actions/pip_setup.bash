# --allow-unsafe for setuptools
pip-compile --generate-hashes --allow-unsafe requirements.in --output-file requirements_lock.txt
