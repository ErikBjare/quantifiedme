dashboard:
	pipenv run python3 build_dashboard.py

test:
	pipenv run python3 -m pytest scripts/habitbull.py
	#make --directory=smartertime2activitywatch test
	make --directory=QSlang test
	make --directory=chatalysis test



jupyter:
	# From: https://stackoverflow.com/a/47296960/965332
	pipenv install --skip-lock ipykernel
	pipenv run bash -c 'python -m ipykernel install --user --name=`basename $$VIRTUAL_ENV`'
