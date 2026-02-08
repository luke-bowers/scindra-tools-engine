# Contributing

## Development setup

The project uses [uv](https://docs.astral.sh/uv/) for fast, reproducible installs. Pip is also supported.

### With uv (recommended)

1. [Install uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Clone the repository and from the repo root run:

   ```bash
   uv sync --extra dev
   ```

   This creates a virtual environment (e.g. `.venv`), installs the package in editable mode, and installs dev dependencies. If `pyproject.toml` or optional deps change, run `uv lock --extra dev` and commit `uv.lock`.

### With pip

1. Clone the repository.
2. Create a virtual environment and install the package in editable mode with dev dependencies:

   ```bash
   pip install -e ".[dev]"
   ```

## Running checks

- **Lint:** `ruff check .` (or `make lint`; with uv: `uv run ruff check .`)
- **Wheelhouse smoke (optional):** For full constraints + offline wheelhouse validation, run `./scripts/smoke_constraints_wheelhouse_local.sh` or `./scripts/smoke_constraints_wheelhouse_local.ps1` from the repo root (requires Python 3.11+ and pip only).
- **Type check:** `mypy src` (or `make type`; with uv: `uv run mypy src`)
- **Tests:** `PYTHONPATH=src pytest` (or `make test`; with uv: `uv run pytest`)

On Windows, run the commands directly if `make` is not available:

```powershell
ruff check .
mypy src
$env:PYTHONPATH="src"; pytest
```

With uv, use `uv run` so the project venv is used:

```powershell
uv run ruff check .
uv run mypy src
uv run pytest
```

## Code style

- Formatting and linting: [Ruff](https://docs.astral.sh/ruff/).
- Type checking: [mypy](https://mypy.readthedocs.io/) (strict mode).

Ensure `make lint`, `make type`, and `make test` pass before opening a pull request.

## Pull requests

1. Open an issue or pick an existing one.
2. Branch from `main`, make your changes, and run lint/type/test.
3. Open a PR against `main`. CI will run the same checks.
