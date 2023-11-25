
server:
	@python aiy_web.py


clear:
	@rm -rf dist build

build: clear
	@python setup.py bdist_wheel

publish: build
	@python -m twine upload dist/*
