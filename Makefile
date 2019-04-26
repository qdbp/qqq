.PHONY: test cover clean_test doc

test:
	pytest

clean_test:
	rm .testmondata	

cover: clean_test
	pytest -c cover

doc: build
	sphinx-build doc doc-out

_build:
	python setup.py build
