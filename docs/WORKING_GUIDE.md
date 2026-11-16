# Daily FYP working guide

A practical reference for Syazani and a fresh coding-agent session. Read [PROJECT_HANDOVER.md](PROJECT_HANDOVER.md) for scientific context and [SETUP.md](SETUP.md) for setup details.

## 1. Which folder should I work in?

Use `fyp-organoid-analysis` as your main project folder. Open this folder in VS Code, Antigravity or your chosen editor. On the original computer its full path is `C:\Users\User\Desktop\FYP\fyp-organoid-analysis`.

The parent FYP folder is the original archive and data source. Keep its recordings, administrative documents, old prototypes and raw collection there. Edit active code in the new repo so changes can be tracked. Do not delete old files merely because they were excluded from GitHub.

## 2. Folder map

| Folder | Purpose | What belongs here | GitHub behaviour |
|---|---|---|---|
| `src/` | Working software | App, UI, model definition, training and inference code | Tracked |
| `models/` | Selected model | Reviewed checkpoint, hash/model card and fitting records | Tracked; no automatic candidate promotion |
| `data/metadata/` | Dataset records | Inventory, annotation register and snapshot hashes | Tracked |
| `data/pilot/` | Local pilot dataset | `images/`, `masks/`, `rois/`, `measurements/` | Payloads ignored; README tracked |
| `data/raw/` | Optional full dataset location | Original full-resolution inputs | Payloads ignored; README tracked |
| `experiments/` | Research record | Experiment plan, configuration, metrics, interpretation and log | Tracked unless separately ignored |
| `results/local/` | Working outputs | Generated predictions, temporary checks and candidate models | Ignored |
| Other `results/` folders | Selected evidence | Reviewed small reports with reproducible provenance | Tracked unless separately ignored |
| `docs/` | Living documentation | Scope, status, architecture, roadmap, setup and this guide | Tracked |
| `docs/meetings/` | Technical supervision notes | Confirmed decisions and action items | Tracked; avoid private conversation |
| `docs/progress/` | Dated progress updates | Completed work, evidence, blockers and next steps | Tracked |
| `docs/literature/` | Reading register | Citations, method notes, limitations and relevance | Tracked |
| `reports/` | Shareable milestone documents | Reviewed reports and presentation exports | Tracked unless separately ignored |
| `scripts/` | Supporting tools | Dataset restore and repository checks | Tracked |
| `tests/` | Behaviour verification | Existing CNN and Fiji integration checks | Tracked |
| `.github/` | GitHub configuration | Automatic checks and issue/PR templates | Tracked |

An ignored folder is not backed up by pushing Git. Keep separate backups for datasets and valuable generated outputs. Files in `reports/`, `experiments/` and most of `docs/` are not automatically private; review them before staging.

## 3. Important source files

| File | Function |
|---|---|
| `src/prototype_app.py` | Starts the localhost upload and measurement server |
| `src/prototype_ui.html` | Browser measurement interface |
| `src/organoid_core.py` | TIFF validation, preprocessing, model inference, polygon and measurements |
| `src/project_paths.py` | Repository-relative locations and environment overrides |
| `src/train_prototype.py` | Fits all available paired masks and writes a candidate |
| `src/train_three.py` | Defines TinyUNet and retains the earlier three-image baseline |
| `src/predict_image.py` | Earlier square-resize baseline inference; not the current app pipeline |
| `src/network_visualizer.py` | CNN inspector server |
| `src/network_visualizer.html` | CNN inspector interface |
| `src/imagej_helper/MeasureOrganoid.java` | Fiji/ImageJ measurement bridge |
| `scripts/import_local_data.py` | Restores manifest-listed pilot files from an original FYP folder |
| `scripts/check_repository.py` | Checks syntax, links, manifests and checkpoint checksum |

Keep Python servers and their HTML files together. Changing a filename or folder may require updating imports, paths and documentation.

## 4. Important root files

- `README.md`: public landing page and overview; update when the project's main behaviour changes.
- `AGENTS.md`: instructions for coding agents; explicitly ask new sessions to read it.
- `CONTRIBUTING.md`: development workflow.
- `requirements.txt`: Python packages required by the prototype.
- `requirements-tested.txt`: versions from the verified local environment, not a universal lockfile.
- `.gitignore`: local files that should not normally be committed. It does not remove previously committed files from history.
- `.gitattributes`: Git text/binary handling and line-ending conventions.

## 5. Open and run the current prototype

From PowerShell on the original computer:

```powershell
cd C:\Users\User\Desktop\FYP\fyp-organoid-analysis
$env:FIJI_HOME = 'C:\Users\User\Downloads\fiji-latest-win64-jdk\Fiji'
& 'C:\Users\User\miniconda3\python.exe' src/prototype_app.py
```

These are the previously verified workstation locations; check they still exist. Open http://127.0.0.1:8765 if the browser does not open. Stop the server with Ctrl+C in its terminal. If the port is occupied, add `--port 8766` and use that port in the browser.

For the CNN inspector, open another terminal:

```powershell
& 'C:\Users\User\miniconda3\python.exe' src/network_visualizer.py
```

Open http://127.0.0.1:8770. The CNN inspector does not perform the Fiji measurement step.

On a fresh machine, follow [SETUP.md](SETUP.md): create an environment, install packages, set `FIJI_HOME` and restore authorised data. Do not assume a command named `python` refers to the environment containing PyTorch. Check with `python -c "import sys; print(sys.executable)"`.

Path overrides supported by `src/project_paths.py`:

| Variable | Purpose | Default |
|---|---|---|
| `FYP_DATA_DIR` | Folder containing images/masks/rois/measurements | `data/pilot` |
| `FYP_MODEL_PATH` | Selected checkpoint | `models/tiny_unet.pt` |
| `FYP_RESULTS_DIR` | Generated output root | `results/local` |
| `FIJI_HOME` | Fiji installation including Java | `tools/Fiji` |

Set these before starting Python. Current ImageJ Java discovery targets Windows; do not claim Linux/macOS support without implementing and testing it.

## 6. Start each work session

1. Open the repository root, not the old `training_pilot` folder.
2. Read the latest progress note and roadmap.
3. Run `git status` and `git branch --show-current`.
4. If your working tree is clean and you need remote updates, use `git pull --ff-only`.
5. Choose one concrete task. For example: verify calibration, review three annotations, fix an upload error or compare two segmentation settings.

If Git reports existing changes, inspect them before switching branches or pulling. Do not discard them to make a command succeed.

## 7. Record different kinds of progress

### After a supervisor meeting

Copy `docs/meetings/TEMPLATE.md` to a dated filename, such as `2026-11-20-scope-review.md`. Record only what actually happened: decisions, action owners, due dates and unresolved questions. Update status/roadmap if the agreed scope changes. Keep recordings and private conversation outside the public repo.

### After reading a paper

Update the register in `docs/literature/README.md` with citation/DOI, dataset, method, validation design, findings, limitations and relevance. A paper's result does not establish the same result on this dataset. Link to lawful sources rather than adding publisher PDFs by default.

### After adding annotations

Preserve original TIFFs and consistent stems across image, mask and ROI. Record annotation source, date, reviewer and status. Check image dimensions, binary mask values and ROI alignment. Keep corrected measurement CSVs separate from originals with a correction explanation. Version and update integrity manifests deliberately after review; the current importer only restores files in the original snapshot.

### After an experiment

Create a descriptive experiment folder, for example:

```text
experiments/exp002-boundary-comparison/
  README.md
  configuration.json
  metrics.csv
```

This is a suggested structure, not a claim that experiment 002 has been run. Record:

- Research question and hypothesis.
- Code commit, exact command, environment and seed.
- Dataset version/hashes and explicit train/validation/test membership.
- Initial checkpoint hash, preprocessing and parameter settings.
- Metrics with units, per-image/group results and failure cases.
- What changed relative to the comparison baseline.
- Conclusion, limitations and keep/reject/follow-up decision.

Add a row to `experiments/log.csv`. Put large generated files under `results/local/`; retain only selected approved evidence in tracked folders. Avoid replacing the selected checkpoint simply because a training score improved.

### At the end of a work session

Copy `docs/progress/TEMPLATE.md` to a dated note. Record completed work, evidence, limitations, blockers and the next action. Update the main README only when the public overview needs to change; it should not become a daily diary.

## 8. Update GitHub: the simple workflow

Saving a file changes your working copy. `git add` selects changes for a commit. `git commit` records a local version. `git push` sends commits to GitHub.

After a small documentation update on `main`:

```powershell
git status
git diff
python scripts/check_repository.py
```

Stage only the paths you intend to publish. Example (use a filename you actually created):

```powershell
git add docs/progress/2026-11-20.md
git diff --cached
git status
git commit -m "Record annotation review progress and next actions"
git push
```

`git diff` shows unstaged edits to tracked files; new untracked files need to be opened directly or reviewed after staging with `git diff --cached`. If a check fails, fix or investigate it before publishing; do not treat the example commands as a script that ignores failures.

To unstage a mistakenly selected file without deleting your edit:

```powershell
git restore --staged path/to/file
```

Avoid `git add .` until you have reviewed every new file. Never use `git add -f` to bypass dataset exclusions just to make an upload work.

## 9. Use a branch for code or experimental changes

Starting from a clean, up-to-date `main`:

```powershell
git switch main
git pull --ff-only
git switch -c feature/calibration-review
```

Make the change, document it and run applicable checks. Stage the specific files, then:

```powershell
git commit -m "Improve calibration validation"
git push -u origin feature/calibration-review
```

On GitHub, open a pull request from the feature branch into `main`. Explain the change and checks, review the diff and merge after the required checks pass. For a solo project, this is useful self-review practice. Your supervisor can inspect the PR but is not automatically asked to approve it.

After merging, with local changes committed:

```powershell
git switch main
git pull --ff-only
```

Do not force-push to resolve a rejected push. First inspect remote changes. If `--ff-only` fails, the histories diverged; ask the coding agent to inspect and reconcile them while preserving your work.

## 10. Which checks should I run?

| Change | Applicable check |
|---|---|
| Documentation, manifests or repository layout | `python scripts/check_repository.py` |
| CNN inspector or model arithmetic | `python tests/verify_network_visualizer.py` |
| Segmentation exports, ImageJ bridge or measurement workflow | `python tests/verify_prototype.py` |
| Model selection or research method | Independent evaluation designed for that research question, plus affected software checks |

Run Python from the correct environment. The integration checks require pilot data and checkpoint; the Fiji check also requires `FIJI_HOME`. They write ignored local outputs. GitHub's automatic workflow runs syntax/structure checks only; a green Actions check does not mean Fiji integration or scientific validation ran in the cloud.

## 11. Continue the research in order

1. Confirm final scope and measurable objectives with the supervisor.
2. Agree the target boundary and annotation protocol.
3. Verify calibration and repair reference-measurement issues with provenance.
4. Expand and review annotations.
5. Define independent splits by specimen/well and, where appropriate, batch; keep related days together.
6. Add split-aware training/evaluation code. The current fitter uses all paired masks.
7. Compare classical and learning baselines on the same frozen evaluation data.
8. Evaluate segmentation and each of the eight measurement errors, including failure cases.
9. Select a candidate based on documented evidence and update its model card/hash.
10. Prepare a reproducible demonstration and reviewed report; consider extensions only after scope/data confirmation.

These are research steps, not predetermined positive outcomes. Record negative results and unresolved questions as progress too.

## 12. Public repo, backups and sharing

The repo is public as of 16 November 2026. Anyone can read or copy committed files. Keep credentials, raw meeting material, CVs, private correspondence and unapproved lab data out of commits. Public visibility does not automatically assign an open-source licence; no licence has been selected.

GitHub is a backup for committed and pushed content, not ignored data or uncommitted work. Back up the original dataset, reviewed annotations and valuable local outputs separately. Check `git status` after pushing and check the new commit and Actions status on GitHub.

If you accidentally commit sensitive material, stop publishing further copies and seek targeted cleanup. Adding a filename to `.gitignore` does not erase its previous commits.

## 13. Returning after a break

Read the latest progress note, `PROJECT_STATUS.md`, `ROADMAP.md` and the handover. Give Codex one clear session task and ask it to read `AGENTS.md`. Verify local data and environment availability before spending time debugging missing-data failures. End the session with an updated progress note and a clear statement of what was tested, committed and pushed.
