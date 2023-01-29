# today
date := $(shell date +%Y-%m-%d)

all: notebooks

notebooks: notebooks/output/Dashboard.html

notebooks/output/Dashboard.html:
	env PERSONAL=false FAST=true make -C notebooks build


dashboard: dashboard/build/dashboard.html

dashboard/build/dashboard.html:
	poetry run python3 dashboard/build_dashboard.py

install:
	poetry install

fmt:
	black src/ tests/

test:
	poetry run python3 -m pytest tests/

typecheck:
	poetry run mypy --ignore-missing-imports --check-untyped-defs src/quantifiedme tests

precommit:
	make fmt
	poetry run pyupgrade --py310-plus src/quantifiedme/**.py
	make typecheck
	make test

jupyter:
	# From: https://stackoverflow.com/a/47296960/965332
	poetry run bash -c 'python -m ipykernel install --user --name=`basename $$VIRTUAL_ENV`'
