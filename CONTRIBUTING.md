# Personal development workflow

1. Open an issue for one research question, bug or milestone.
2. Create a short branch: `git switch -c feature/short-description`.
3. Make the change; update the relevant experiment/progress record.
4. Format and check: `ruff format .`, `ruff check .`, `python -m pytest`, plus the applicable integration checks from `docs/SETUP.md`.
5. Review `git status` and `git diff` before staging named files. Avoid adding raw data or private material.
6. Commit a clear message, push the branch and open a pull request. Self-review and merge after checks pass.

## Code conventions

- One responsibility per module; see `docs/ARCHITECTURE.md` for where things belong.
- Every module and public function has a docstring that says what it does and what it returns.
- No absolute workstation paths: import locations from `src/project_paths.py`.
- Named constants instead of unexplained numbers.
- A new behaviour gets a unit test in `tests/test_core.py` when it can be tested without lab data.
- Record notable changes in `CHANGELOG.md`.

## Reporting

Every supervisor update should distinguish completed work, evidence, limitations, decisions needed and next actions. Do not describe model fitting scores as test accuracy. Keep original data immutable; derived labels and corrected measurements need a separate provenance record.
