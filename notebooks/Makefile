.PHONY: build build-personal set-testing set-personal

build:
	poetry run jupyter nbconvert 'QuantifiedMe - Dashboard.ipynb' --execute --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=600

build-personal:
	make set-personal
	make build
	make set-testing

set-testing:
	sed -i 's/personal = True/personal = False/' 'QuantifiedMe - Dashboard.ipynb'  # Set personal = False, in case it was accidentally comitted with `personal = True`

set-personal:
	sed -i 's/personal = False/personal = True/' 'QuantifiedMe - Dashboard.ipynb'
