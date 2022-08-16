SHELL=/bin/bash

SRC_FILES := $(wildcard *.ipynb personal/*.ipynb)
OUT_FILES := $(patsubst %.ipynb,output/%.html,$(SRC_FILES))

# today
TODAY := $(shell date +%Y-%m-%d)
MONTH := $(shell date +%Y-%m)
YEAR := $(shell date +%Y)

build: $(OUT_FILES) output/index.html

clean-cache:
	rm -rf .cache
	# if personal/.cache exists, delete it
	[ -d personal/.cache ] && rm -rf personal/.cache

.PHONY: output/index.html
output/index.html:
	@echo -e "<html><head><title>QuantifiedMe Index</title></head>" > $@
	@echo -e "<body><h2>Index of notebooks</h2><table>" >> $@
	@echo -e "<thead><th>Notebook</th><th>Age</th><th>Errors</th><th>Warnings</th><th>Last updated</th></thead>" >> $@
	@# if built then include modifed time
	@# TODO: Compute AGE with JS from LAST_UPDATED instead
	@for i in $(patsubst output/%,%,$(patsubst output/personal/%,%,$(OUT_FILES))); do \
		echo -e '<tr>' >> $@; \
		echo -e "<td><a href='$$i' target="_blank">$(basename $$i)</a></td>" >> $@; \
		if [ -f output/$$i ]; then \
			LAST_UPDATED=$$(date +%s -r output/$$i); \
			AGE=$$(($$(date +%s) - $$LAST_UPDATED)); \
			echo -e "<td>$$AGE s ago</td>" >> $@; \
			ERRORS=$$(grep 'error' --ignore-case output/$$i | wc -l); \
			ERRORS=$$(($$ERRORS - 20)); \
			if [ $$ERRORS -gt 0 ]; then \
				echo -e "<td><font color='red'>$$ERRORS</font></td>" >> $@; \
			else \
				echo -e "<td>$$ERRORS</td>" >> $@; \
			fi; \
			WARNINGS=$$(grep 'warning' --ignore-case output/$$i | wc -l); \
			WARNINGS=$$(($$WARNINGS - 160)); \
			if [ $$WARNINGS -gt 0 ]; then \
				echo -e "<td><font color='orange'>$$WARNINGS</font></td>" >> $@; \
			else \
				echo -e "<td>$$WARNINGS</td>" >> $@; \
			fi; \
			echo -e "<td>$$(date '+%Y-%m-%d %H:%M' -r output/$$i)</td>" >> $@; \
		else \
			echo -e "<td>Not built</td>" >> $@; \
		fi; \
		echo -e '</tr>' >> $@; \
	done
	@echo -e "</ul></body></html>" >> $@
	@echo "Built index.html"

output/%.html: %.ipynb set-env
	poetry run jupyter nbconvert $< \
		--output-dir output --to=html \
		--execute --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=1200 \
		--allow-errors #--no-input
	@make output/index.html  # update index

output/%/index.md: %.ipynb
	poetry run jupyter nbconvert $< \
		--output-dir output --to md \
		--execute --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=600

# WIP
output/%/index.html: %.ipynb
	make output/$(notdir $<)

# WIP
output/%/$TODAY/index.html: %.ipynb
	# TODO: Set query range via env variable
	make output/$(notdir $<)/$TODAY

precommit:
	poetry run jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace *.ipynb

clean:
	rm -r output

# Notebook settings

.PHONY: set-env
set-env:
ifeq ($(PERSONAL),true)
	make set-personal
else
	make set-testing
endif
# fall back to fast mode
ifeq ($(FAST),true)
	make set-fast
else ifeq ($(FAST),false)
	make set-slow
else
	make set-fast
endif

.PHONY: set-fast
set-fast:
	sed -i 's/fast = False/fast = True/' $(SRC_FILES)

.PHONY: set-slow
set-slow:
	sed -i 's/fast = True/fast = False/' $(SRC_FILES)

.PHONY: set-testing
set-testing:
	sed -i 's/personal = True/personal = False/' $(SRC_FILES)  # Set personal = False, in case it was accidentally comitted with `personal = True`

.PHONY: set-personal
set-personal:
	sed -i 's/personal = False/personal = True/' $(SRC_FILES)

debug:
	echo $(SRC_FILES)
	echo $(OUT_FILES)
