.PHONY: build build-personal

DASHBOARD_SRC=Dashboard.ipynb
DASHBOARD=output/Dashboard.html

build: output/Dashboard.html output/personal/All.html

output/%.html: %.ipynb
	poetry run jupyter nbconvert $^ --output-dir output --execute --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=600

precommit:
	poetry run jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace *.ipynb

build-personal: $(DASHBOARD)

build-personal-slow:
	make set-personal
	make set-slow
	make build-personal
	make set-testing

build-personal-fast:
	make set-personal
	make set-fast
	make build-personal
	make set-slow
	make set-testing

clean:
	rm -r output

# Notebook settings

.PHONY: set-fast
set-fast:
	sed -i 's/fast = False/fast = True/' $(DASHBOARD_SRC)

.PHONY: set-slow
set-slow:
	sed -i 's/fast = True/fast = False/' $(DASHBOARD_SRC)

.PHONY: set-testing
set-testing:
	sed -i 's/personal = True/personal = False/' $(DASHBOARD_SRC)  # Set personal = False, in case it was accidentally comitted with `personal = True`

.PHONY: set-personal
set-personal:
	sed -i 's/personal = False/personal = True/' $(DASHBOARD_SRC)
