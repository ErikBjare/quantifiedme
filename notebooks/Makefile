DASHBOARD_SRC=Dashboard.ipynb
DASHBOARD=output/Dashboard.html

SRC_FILES := $(wildcard *.ipynb personal/*.ipynb)
OUT_FILES := $(patsubst %.ipynb,output/%.html,$(SRC_FILES))

build: $(OUT_FILES)

output/%.html: %.ipynb
ifeq ($(PERSONAL),true)
	make set-personal FILE=$^
else
	make set-testing FILE=$^
endif
ifeq ($(FAST),true)
	make set-fast FILE=$^
else
	make set-slow FILE=$^
endif
	poetry run jupyter nbconvert $^ --output-dir output --to=html --execute --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=600

precommit:
	poetry run jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace *.ipynb

clean:
	rm -r output

# Notebook settings

.PHONY: set-fast
set-fast:
	sed -i 's/fast = False/fast = True/' $(FILE)

.PHONY: set-slow
set-slow:
	sed -i 's/fast = True/fast = False/' $(FILE)

.PHONY: set-testing
set-testing:
	sed -i 's/personal = True/personal = False/' $(FILE)  # Set personal = False, in case it was accidentally comitted with `personal = True`

.PHONY: set-personal
set-personal:
	sed -i 's/personal = False/personal = True/' $(FILE)

debug:
	echo $(SRC_FILES)
	echo $(OUT_FILES)
