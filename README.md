# scindra-engine

Scindra tools engine.

## Requirements

- Python 3.11+

## Installation

```bash
pip install scindra-engine
```

For development, the project uses [uv](https://docs.astral.sh/uv/) for fast, reproducible installs. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Usage

```bash
scindra-engine --version
```

## How to cut a release

1. **Bump version**  
   Edit `version` in [pyproject.toml](pyproject.toml) (e.g. `0.1.0` → `0.2.0`). The package reads this as the single source of truth.

2. **Validate locally**  
   Run the smoke script so lint, typecheck, tests, build, twine check, and a local wheel install are exercised before tagging:
   - **mac/Linux:** `./scripts/smoke_release_local.sh`
   - **Windows:** `./scripts/smoke_release_local.ps1`  
   Requires [uv](https://docs.astral.sh/uv/) and Python 3.11.

3. **Tag**  
   Create a tag that matches the version in `pyproject.toml` (with a `v` prefix), e.g.:
   ```bash
   git tag v0.2.0
   ```

4. **Push the tag**  
   Pushing the tag triggers the release workflow (lint, typecheck, tests, version check, build, twine check, publish to PyPI, create GitHub Release):
   ```bash
   git push origin v0.2.0
   ```

5. **Where artifacts appear**  
   - **GitHub:** The tag’s [Releases](https://docs.github.com/en/repositories/releasing-projects-on-github) page will have the sdist and wheel attached.
   - **PyPI:** The package will be published to [PyPI](https://pypi.org/project/scindra-engine/) once the workflow completes.

### PyPI publishing (Trusted Publishing / OIDC)

The workflow uses [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) (OIDC) when no token is configured, so you don’t need to store a PyPI API token in GitHub.

**One-time setup on PyPI:**

1. Open your project on PyPI → **Settings** → **Publishing** → **Add a new trusted publisher**.
2. Choose **GitHub Actions**.
3. Set **Owner** and **Repository** to this repo.
4. Set **Workflow name** to `release.yml`.
5. Optionally set **Environment** (e.g. `pypi`) if you use a GitHub environment for approvals.

The workflow job already has `id-token: write` and uses `pypa/gh-action-pypi-publish` without username/password when OIDC is configured.

**Fallback (API token):**  
If you prefer not to use Trusted Publishing, add a repository secret `PYPI_API_TOKEN` with a PyPI API token, and in the workflow pass it to the publish step (e.g. `password: ${{ secrets.PYPI_API_TOKEN }}`, `username: __token__`). See the [action’s documentation](https://github.com/pypa/gh-action-pypi-publish) for details.

## License

[Apache-2.0](LICENSE)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).
