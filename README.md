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

## Smoke tests

The `smoke_latest` script provides an adaptive smoke test that detects available CLI capabilities and exercises the most end-to-end path available. It uses CLI commands exclusively and produces artifacts in `out/smoke_latest/<timestamp>/`.

**Basic usage:**
```bash
bash scripts/smoke_latest.sh
```

**With optional real video testing:**
```bash
VIDEO_PATH=/path/to/video.mp4 bash scripts/smoke_latest.sh
```

**Strict mode (fail on real video errors):**
```bash
STRICT_REAL_VIDEO=1 VIDEO_PATH=/path/to/video.mp4 bash scripts/smoke_latest.sh
```

The script automatically adapts to available capabilities:
- **Baseline UX** (always): `engine-info`, `probe`, `extract-frames`
- **E3 Config UX** (if available): `init-config`, `validate-config`
- **Track-Centroid** (if available): runs on synthetic fixtures, validates outputs including `overlay.mp4` and `heatmap.png` when supported
- **E6.2+ features** (if available): validates `manifest.json` and `support_bundle.zip`
- **Auto-Setup** (if available): tests arena and assay detection
- **Batch** (if available): tests batch processing

**Windows PowerShell:**
```powershell
.\scripts\smoke_latest.ps1
```

## How to cut a release

1. **Bump version**  
   Edit `version` in [pyproject.toml](pyproject.toml) (e.g. `0.1.0` → `0.2.0`). The package reads this as the single source of truth.

2. **Validate locally**  
   Run the smoke script so lint, typecheck, tests, build, twine check, and a local wheel install are exercised before tagging:
   - **mac/Linux:** `./scripts/smoke_release_local.sh`
   - **Windows:** `./scripts/smoke_release_local.ps1`  
   Requires [uv](https://docs.astral.sh/uv/) and Python 3.11. For full wheelhouse/offline validation (constraints + offline install), run `./scripts/smoke_constraints_wheelhouse_local.sh` or `./scripts/smoke_constraints_wheelhouse_local.ps1` (requires Python 3.11+ and pip only).

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
   - **GitHub:** The tag’s [Releases](https://docs.github.com/en/repositories/releasing-projects-on-github) page will have the sdist, wheel, and `constraints-desktop.txt` attached.
   - **PyPI:** The package will be published to [PyPI](https://pypi.org/project/scindra-engine/) once the workflow completes.

### constraints-desktop.txt and Desktop Pro wheelhouse

`constraints-desktop.txt` is a pip constraints file generated from the built wheel’s resolved environment: after installing the wheel in a temporary venv, we run `pip freeze` and strip `pip`, `setuptools`, `wheel`, and the project itself. The result pins exact versions of the runtime dependencies only, so `pip download -c constraints-desktop.txt <wheel>` uses the same dependency versions every time without conflicting with the local wheel.

**Desktop Pro** bundles an offline wheelhouse per platform. It downloads `constraints-desktop.txt` from the GitHub Release and uses it with `pip download -c constraints-desktop.txt ...` when building the wheelhouse, so the same dependency versions are used every time. The file is not published to PyPI; it is only attached to the GitHub Release.

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
