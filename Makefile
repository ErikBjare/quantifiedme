SRCDIRS := src/quantifiedme tests
SRCFILES := $(shell find $(SRCDIRS) -name '*.py')

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

lint:
	poetry run ruff $(SRCDIRS)

fmt:
	poetry run ruff --fix $(SRCDIRS)
	poetry run pyupgrade --py310-plus $(SRCFILES) --exit-zero-even-if-changed
	poetry run black $(SRCDIRS)

test:
	poetry run python3 -m pytest -v tests/ --cov=quantifiedme --durations=5

typecheck:
	poetry run mypy --ignore-missing-imports --check-untyped-defs $(SRCDIRS)

precommit: fmt typecheck test

jupyter:
	# From: https://stackoverflow.com/a/47296960/965332
	poetry run bash -c 'python -m ipykernel install --user --name=`basename $$VIRTUAL_ENV`'
