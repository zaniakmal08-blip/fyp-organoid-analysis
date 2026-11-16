# Changelog

Notable changes to the code and repository. Research progress is recorded in `docs/progress/`; experiments in `experiments/log.csv`.

## 2026-11-16 — Code restructure

No change to model weights, preprocessing or measurement results; outputs were verified identical to the previously recorded evidence.

### Added
- `src/model.py`, `src/preprocessing.py` and `src/imagej_bridge.py`, split out of `organoid_core.py` and `train_three.py`.
- Docstrings for every module and the main public functions.
- `tests/test_core.py`: unit tests that run without lab data or Fiji.
- `pyproject.toml` (Ruff and pytest configuration), `requirements-dev.txt`, `.editorconfig`.
- Shared run configurations in `.vscode/` for VS Code and Antigravity.
- GitHub Actions now runs the unit tests and style checks.
- Detailed `docs/ARCHITECTURE.md`; troubleshooting section in `docs/SETUP.md`.

### Changed
- All Python formatted with Ruff (100-column lines).
- `TinyUNet` now lives in `model.py`. `organoid_core` still re-exports the helper functions, so existing imports keep working.
- The apps read example images from `project_paths.DATA` instead of importing the legacy training script.
- README rewritten as a project landing page.

### Fixed
- `predict_image.py` no longer contains a hard-coded personal file path; `--image` and `--model` are required arguments and previews are written under `results/local/`.

## 2026-11-16 — Repository created

- Curated the working prototype, checkpoint, data manifests and documentation from the original FYP folder (see `docs/MIGRATION.md`).
