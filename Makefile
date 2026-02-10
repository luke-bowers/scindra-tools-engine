.PHONY: setup lint type test build

setup:
	uv sync --extra dev --frozen

lint:
	uv run ruff check .

type:
	uv run mypy src

ifeq ($(OS),Windows_NT)
test:
	set PYTHONPATH=src && uv run pytest
else
test:
	PYTHONPATH=src uv run pytest
endif

build:
	uv run python -m build
