# Agent instructions for this FYP repository

## Read first

1. Read `docs/PROJECT_HANDOVER.md` for the current scientific and implementation context.
2. Read `docs/WORKING_GUIDE.md` for folder ownership, commands and Git workflow.
3. Read `docs/PROJECT_STATUS.md`, `docs/ROADMAP.md` and `data/DATA_NOTES.md` before proposing experiments or interpreting results.
4. Inspect `git status --short`, the current branch and relevant owning code before editing. Preserve uncommitted user work.

These files support a fresh Codex session in Antigravity or another editor. Do not assume access to previous chat history or that an editor automatically loaded this file. The user's current instructions take precedence.

## Project boundaries

- This is Syazani Akmal's individual organoid-image-analysis FYP, not the team's Kaggle solar-filament project.
- The existing deliverable is a local segmentation and Fiji/ImageJ morphometry prototype plus a CNN inspector.
- The six-image model has no independent validation. Never call training Dice test accuracy or evidence of biological reliability.
- Prediction, halo/differentiation analysis and microscope capture are possible directions, not approved or implemented deliverables.
- Keep source images, original annotations and reference CSVs immutable. Store reviewed corrections separately with provenance.
- Do not match the NC004 workbook to toxicity TIFFs using similar well names; the correspondence is unresolved.

## Implementation conventions

- Run commands from the repository root. Use `src/project_paths.py` and environment overrides; do not reintroduce absolute workstation paths into code.
- Keep the HTML interfaces next to their Python servers. TinyUNet is still defined in `src/train_three.py`, which the current pipeline imports.
- The selected checkpoint is `models/tiny_unet.pt`. Training candidates belong in `results/local/` and must not silently replace it.
- Full morphometry currently requires Windows Fiji with bundled Java. Local servers bind to localhost; do not expose them remotely as part of routine development.
- Default training fits all paired masks; it does not implement a held-out split. Build and verify split support before claiming independent evaluation.
- For reproducibility, record data hashes, model hash, code revision, seed, configuration, split and command for each experiment.

## Checks and publication

- Run `python scripts/check_repository.py` for documentation/structure changes.
- Run applicable existing integration checks for code changes; see `docs/SETUP.md`. Do not run training merely to verify a documentation edit.
- Report what was tested and what was unavailable. Software checks do not replace scientific validation.
- This repository became public on 16 November 2026. Review staged content before publishing. Do not force-add ignored datasets, credentials, recordings, transcripts or administrative/personal documents.
- Git ignores pilot payloads, raw images and `results/local/`; those need separate backups.
- Commit/push when requested or clearly within the active task. Do not assume every future request authorises publication, history rewriting, branch deletion or model promotion.
- Update the handover/status when scope, paths, model selection or verified findings change. Do not invent dates, supervisor decisions or results to fill templates.
