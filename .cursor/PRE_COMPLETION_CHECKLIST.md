# Pre-Completion Checklist

**MANDATORY: Run this checklist before marking ANY task as complete**

## Required Checks (from .cursor/cursorrules)

1. **Release Smoke Script** (MANDATORY)
   - [ ] Run `.\scripts\smoke_release_local.ps1` (Windows) or `./scripts/smoke_release_local.sh` (Linux/Mac)
   - [ ] Verify ALL steps pass: lint, type check, tests, build, twine check, wheel install, version check
   - [ ] Fix any failures before proceeding

2. **Type Checking**
   - [ ] Run `python -m mypy src` (or `uv run mypy src`)
   - [ ] Verify: "Success: no issues found"

3. **Linting**
   - [ ] Run `python -m ruff check .` (or `uv run ruff check .`)
   - [ ] Verify: "All checks passed!"

4. **Tests**
   - [ ] Run `python -m pytest` (or `uv run pytest`)
   - [ ] Verify: All tests pass

5. **Code Review Checklist** (from cursorrules)
   - [ ] Is the change deterministic?
   - [ ] Are side effects contained to IO modules?
   - [ ] Are types accurate and mypy-friendly?
   - [ ] Are tests added/updated and not flaky?
   - [ ] Are public functions documented?
   - [ ] Does the change avoid new dependencies (or are they justified)?
   - [ ] Do errors remain actionable?
   - [ ] Has the release smoke script been run and passed? ⚠️ **CRITICAL**

## Workflow

**Before saying "task complete" or "done":**
1. Read `.cursor/cursorrules` section "RELEASE SMOKE GATE"
2. Run the release smoke script
3. If it fails, fix issues and re-run until it passes
4. Only then mark tasks as complete

## Notes

- The release smoke script is the **single source of truth** for completion
- Individual checks (mypy, ruff, pytest) are included in the smoke script
- Do NOT skip the smoke script even if individual checks pass
- The smoke script validates the entire release pipeline, not just code correctness
