
test:
	pipenv run python3 -m pytest scripts/habitbull.py
	#make --directory=smartertime2activitywatch test
	make --directory=QSlang test
	make --directory=chatalysis test

