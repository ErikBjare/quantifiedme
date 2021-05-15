all: notebooks

notebooks: notebooks/output/Dashboard.html

notebooks/output/Dashboard.html:
	env PERSONAL=true FAST=true make -C notebooks build


dashboard: dashboard/build/dashboard.html

dashboard/build/dashboard.html:
	poetry run python3 dashboard/build_dashboard.py

install:
	poetry install

fmt:
	black src/ tests/

test:
	poetry run python3 -m pytest tests/

jupyter:
	# From: https://stackoverflow.com/a/47296960/965332
	poetry run pip3 install ipykernel
	poetry run bash -c 'python -m ipykernel install --user --name=`basename $$VIRTUAL_ENV`'
