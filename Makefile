.PHONY: lint type test
lint:
	ruff check .
type:
	mypy src
test:
	PYTHONPATH=src pytest
