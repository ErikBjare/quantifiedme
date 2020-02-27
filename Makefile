dashboard: dashboard/build/dashboard.html

dashboard/build/dashboard.html:
	poetry run python3 dashboard/build_dashboard.py

install:
	poetry install

test:
	poetry run python3 -m pytest scripts/habitbull.py
	#make --directory=smartertime2activitywatch test
	make --directory=QSlang test
	make --directory=chatalysis test

build-notebooks:
	sed -i 's/personal = True/personal = False/' 'QuantifiedMe - Dashboard.ipynb'  # Set personal = False, in case it was accidentally comitted with `personal = True`
	poetry run jupyter nbconvert 'QuantifiedMe - Dashboard.ipynb' --execute --ExecutePreprocessor.kernel_name=python3

jupyter:
	# From: https://stackoverflow.com/a/47296960/965332
	poetry run pip3 install ipykernel
	poetry run bash -c 'python -m ipykernel install --user --name=`basename $$VIRTUAL_ENV`'
