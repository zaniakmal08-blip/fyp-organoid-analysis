# Progress — 2026-11-16: code restructure

## Objective for this period

Make the source code readable and maintainable so the project can be continued in any editor and reviewed by others, without changing what the prototype computes.

## Completed work and evidence

- Split the pipeline into single-purpose modules (`model.py`, `preprocessing.py`, `imagej_bridge.py`, `organoid_core.py`) and documented each one.
- Formatted all Python consistently and added lint/format configuration.
- Added 14 unit tests that need no lab data; GitHub Actions now runs them.
- Added shared editor run configurations, a detailed architecture document and a setup troubleshooting table.
- Re-ran the existing integration checks after the change; see [VERIFICATION.md](../VERIFICATION.md).

## Results and limitations

Predicted measurements for the three reference images are identical to the values recorded in `results/verification-2026-11-16/verification.json`, and the CNN arithmetic check reports the same maximum difference (about 4.16e-6). This is an engineering change only: no new annotations, training or accuracy evidence. Training itself was not re-run.

## Blockers / supervisor decisions

Unchanged — final scope, annotation boundary, calibration and evaluation groups still need confirmation (see [PROJECT_STATUS.md](../PROJECT_STATUS.md)).

## Next actions

See "Next up" in [ROADMAP.md](../ROADMAP.md).
